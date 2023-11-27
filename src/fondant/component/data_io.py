import logging
import os
import typing as t
from collections import defaultdict

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client

from fondant.core.component_spec import ComponentSpec
from fondant.core.manifest import Manifest

logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "id"


class DataIO:
    def __init__(self, *, manifest: Manifest, component_spec: ComponentSpec) -> None:
        self.manifest = manifest
        self.component_spec = component_spec


class DaskDataLoader(DataIO):
    def __init__(
        self,
        *,
        manifest: Manifest,
        component_spec: ComponentSpec,
        input_partition_rows: t.Optional[int] = None,
    ):
        super().__init__(manifest=manifest, component_spec=component_spec)
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
        dataframe = None
        field_mapping = defaultdict(list)

        # Add index field to field mapping to guarantee start reading with the index dataframe
        field_mapping[self.manifest.get_field_location(DEFAULT_INDEX_NAME)].append(
            DEFAULT_INDEX_NAME,
        )

        for field_name in self.component_spec.consumes:
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

        dataframe = self.partition_loaded_dataframe(dataframe)

        logging.info(f"Columns of dataframe: {list(dataframe.columns)}")

        return dataframe


class DaskDataWriter(DataIO):
    def __init__(
        self,
        *,
        manifest: Manifest,
        component_spec: ComponentSpec,
    ):
        super().__init__(manifest=manifest, component_spec=component_spec)

    def write_dataframe(
        self,
        dataframe: dd.DataFrame,
        dask_client: t.Optional[Client] = None,
    ) -> None:
        columns_to_produce = [
            column_name for column_name, field in self.component_spec.produces.items()
        ]

        dataframe.index = dataframe.index.rename(DEFAULT_INDEX_NAME)

        # validation that all columns are in the dataframe
        self.validate_dataframe_columns(dataframe, columns_to_produce)

        dataframe = dataframe[columns_to_produce]
        write_task = self._write_dataframe(dataframe)

        with ProgressBar():
            logging.info("Writing data...")
            dd.compute(write_task, scheduler=dask_client)

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

    def _write_dataframe(self, dataframe: dd.DataFrame) -> dd.core.Scalar:
        """Create dataframe writing task."""
        location = (
            f"{self.manifest.base_path}/{self.manifest.pipeline_name}/"
            f"{self.manifest.run_id}/{self.component_spec.component_folder_name}"
        )

        schema = {
            field.name: field.type.value
            for field in self.component_spec.produces.values()
        }
        return self._create_write_task(dataframe, location=location, schema=schema)

    @staticmethod
    def _create_write_task(
        dataframe: dd.DataFrame,
        *,
        location: str,
        schema: t.Dict[str, str],
    ) -> dd.core.Scalar:
        """
        Creates a delayed Dask task to upload the given DataFrame to the remote storage location
         specified in the manifest.

        Args:
            dataframe: The DataFrame to be uploaded.
            location: the location to write the subset to
            schema: the schema of the dataframe to write

        Returns:
             A delayed Dask task that uploads the DataFrame to the remote storage location when
              executed.
        """
        write_task = dd.to_parquet(
            dataframe,
            location,
            schema=schema,
            overwrite=False,
            compute=False,
        )
        logging.info(f"Creating write task for: {location}")
        return write_task
