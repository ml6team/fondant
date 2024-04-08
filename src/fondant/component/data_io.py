import logging
import os
import typing as t
from collections import defaultdict

import dask.dataframe as dd
import dask.distributed
import fsspec
import pyarrow as pa
from dask.distributed import as_completed

from fondant.core.component_spec import OperationSpec
from fondant.core.manifest import Manifest

logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "id"


class DataIO:
    def __init__(self, *, manifest: Manifest, operation_spec: OperationSpec) -> None:
        self.manifest = manifest
        self.operation_spec = operation_spec


class DaskDataLoader(DataIO):
    def __init__(
        self,
        *,
        manifest: Manifest,
        operation_spec: OperationSpec,
        input_partition_rows: t.Optional[int] = None,
    ):
        super().__init__(manifest=manifest, operation_spec=operation_spec)
        self.input_partition_rows = input_partition_rows

    def partition_loaded_dataframe(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Function that partitions the loaded dataframe depending on its partitions and the available
        workers
        Returns:
            The partitioned dataframe.
        """
        n_workers: int = os.cpu_count()  # type: ignore

        if self.input_partition_rows is None:
            n_partitions = dataframe.npartitions
            if n_partitions < n_workers:  # type: ignore
                logger.info(
                    f"The number of partitions of the input dataframe is {n_partitions}. The "
                    f"available number of workers is {n_workers}.",
                )
                dataframe = dataframe.repartition(npartitions=n_workers)
                logger.info(
                    f"Repartitioning the data to {n_workers} partitions before processing"
                    f" to maximize worker usage",
                )

        elif self.input_partition_rows >= 1:
            # Only load the index column to trigger a faster compute of the rows
            total_rows = len(dataframe.index)
            # +1 to handle any remainder rows
            n_partitions = (total_rows // self.input_partition_rows) + 1
            dataframe = dataframe.repartition(npartitions=n_partitions)
            logger.info(
                f"Total number of rows is {total_rows}.\n"
                f"Repartitioning the data from {dataframe.partitions} partitions to have"
                f" {n_partitions} such that the number of partitions per row is approximately"
                f"{self.input_partition_rows}",
            )
            if n_partitions < n_workers:
                logger.warning(
                    "Setting the `input partition rows` has caused the system to not utilize"
                    f" all available workers {n_partitions} out of {n_workers} are used.",
                )

        else:
            msg = (
                f"{self.input_partition_rows} is not a valid value for the 'input_partition_rows' "
                f"parameter. It should be a number larger than 0 to indicate the number of "
                f"expected rows per partition, or None to let Fondant optimize the number of "
                f"partitions based on the number of available workers."
            )
            raise ValueError(
                msg,
            )

        return dataframe

    def load_dataframe(self) -> dd.DataFrame:
        """
        Function that loads the subsets defined in the component spec as a single Dask dataframe for
          the user.

        Returns:
            The Dask dataframe with all columns defined in the manifest field mapping
        """
        dataframe: t.Optional[dd.DataFrame] = None
        field_mapping = defaultdict(list)

        # Add index field to field mapping to guarantee start reading with the index dataframe
        field_mapping[self.manifest.get_field_location(DEFAULT_INDEX_NAME)].append(
            DEFAULT_INDEX_NAME,
        )

        for field_name in self.operation_spec.consumes_from_dataset:
            location = self.manifest.get_field_location(field_name)
            field_mapping[location].append(field_name)

        for location, fields in field_mapping.items():
            if DEFAULT_INDEX_NAME in fields:
                fields.remove(DEFAULT_INDEX_NAME)

            partial_df = dd.read_parquet(
                location,
                columns=fields,
                index=DEFAULT_INDEX_NAME,
                calculate_divisions=True,
            )

            if dataframe is None:
                # ensure that the index is set correctly and divisions are known.
                dataframe = partial_df
            else:
                dataframe = dataframe.merge(
                    partial_df,
                    how="left",
                    left_index=True,
                    right_index=True,
                )

        if dataframe is None:
            msg = "No data could be loaded"
            raise RuntimeError(msg)

        if (
            consumes_mapping := self.operation_spec.operation_consumes_to_dataset_consumes
        ):
            dataframe = dataframe.rename(
                columns={
                    v: k for k, v in consumes_mapping.items() if isinstance(v, str)
                },
            )

        dataframe = self.partition_loaded_dataframe(dataframe)

        logging.info(f"Columns of dataframe: {list(dataframe.columns)}")

        return dataframe


class DaskDataWriter(DataIO):
    def __init__(
        self,
        *,
        manifest: Manifest,
        operation_spec: OperationSpec,
    ):
        super().__init__(manifest=manifest, operation_spec=operation_spec)

    def write_dataframe(
        self,
        dataframe: dd.DataFrame,
    ) -> None:
        dataframe.index = dataframe.index.rename(DEFAULT_INDEX_NAME)

        # validation that all columns are in the dataframe
        expected_columns = list(self.operation_spec.operation_produces)
        self.validate_dataframe_columns(dataframe, expected_columns)

        dataframe = dataframe[expected_columns]
        if produces_mapping := self.operation_spec._mappings["produces"]:
            dataframe = dataframe.rename(
                columns={
                    k: v for k, v in produces_mapping.items() if isinstance(v, str)
                },
            )
        self._write_dataframe(dataframe)

    @staticmethod
    def validate_dataframe_columns(dataframe: dd.DataFrame, columns: t.List[str]):
        """Validates that all columns are available in the dataset."""
        missing_fields = []
        for col in columns:
            if col not in dataframe.columns:
                missing_fields.append(col)

        if missing_fields:
            msg = (
                f"Fields {missing_fields} defined in output dataset "
                f"but not found in dataframe"
            )
            raise ValueError(
                msg,
            )

    def _write_dataframe(self, dataframe: dd.DataFrame) -> None:
        """Create dataframe writing task."""
        output_location_path = self.manifest.index.location

        if not output_location_path:
            msg = "No output location determined. Can not export the dataset."
            raise ValueError(msg)

        # Create directory the dataframe will be written to, since this is not handled by Pandas
        # `to_parquet` method.
        protocol = fsspec.utils.get_protocol(output_location_path)
        fs = fsspec.get_filesystem_class(protocol)
        fs().makedirs(output_location_path)

        schema = {
            field.name: field.type.value
            for field in self.operation_spec.produces_to_dataset.values()
        }

        # The id needs to be added explicitly since we will convert this to a PyArrow schema
        # later and use it in the `pandas.to_parquet` method.
        try:
            index_type = pa.from_numpy_dtype(dataframe.index.dtype)
        except pa.lib.ArrowNotImplementedError:
            # The dtype of the index is `np._object`. Fall back on string instead.
            logging.warning(
                "Failed to infer dtype of index column, falling back to `string`. "
                "Specify the dtype explicitly to prevent this.",
            )
            index_type = pa.string()

        schema.update(
            {
                "id": index_type,
            },
        )

        # Convert to delayed since computing a dataframe tries to return the complete result,
        # keeping references to all completed tasks, and preventing release of memory.
        # https://distributed.dask.org/en/stable/memory.html#difference-with-dask-compute
        # https://dask.discourse.group/t/improving-pipeline-resilience-when-using-to-parquet-and-preemptible-workers/2141
        to_parquet_tasks = [
            d.to_parquet(
                os.path.join(output_location_path, f"part.{i}.parquet"),
                schema=pa.schema(list(schema.items())),
                index=True,
            )
            for (i, d) in enumerate(dataframe.to_delayed())
        ]

        client: dask.distributed.Client = dask.distributed.get_client()
        futures = client.compute(to_parquet_tasks)

        # As each future completes, release it so the memory can be reclaimed
        for future in as_completed(futures):
            future.result()
            future.release()
