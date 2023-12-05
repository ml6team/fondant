import logging
import os
import typing as t
from collections import defaultdict

import dask.dataframe as dd
import pyarrow as pa
from dask.diagnostics import ProgressBar
from dask.distributed import Client

from fondant.core.component_spec import ComponentSpec
from fondant.core.manifest import Manifest
from fondant.core.schema import Field, ProducesType, Type

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
        consumes: t.Optional[t.Dict[str, str]] = None,
    ):
        super().__init__(manifest=manifest, component_spec=component_spec)
        self.input_partition_rows = input_partition_rows
        self.consumes = consumes
        # fmt: off
        self.non_generic_consumes, self.generic_consumes = self._resolve_custom_consumes()
        # fmt: on

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

    def _resolve_custom_consumes(self) -> t.Tuple[t.Dict[str, str], t.Dict[str, str]]:
        """
        Function that resolves the custom consumes to get generic and non-generic produces fields.

        Returns:
            The generic and non-generic consumes fields.
        """
        generic_consumes = {}
        non_generic_consumes = {}

        if self.consumes is not None:
            for dataset_field, component_field in self.consumes.items():
                if component_field in self.component_spec.consumes:
                    non_generic_consumes[dataset_field] = component_field
                elif self.component_spec.is_consumes_generic:
                    generic_consumes[dataset_field] = component_field
                else:
                    msg = (
                        f"The component spec is not generic and the dataset field"
                        f" {dataset_field} is not defined in the component spec."
                    )
                    raise ValueError(msg)

        return non_generic_consumes, generic_consumes

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

        inverted_non_generic_consumes = {
            v: k for k, v in self.non_generic_consumes.items()
        }

        for original_field_name in self.component_spec.consumes:
            # Remap the field name if it is a non-generic field and is defined in the custom
            # consumes
            remapped_field_name = original_field_name
            if original_field_name in inverted_non_generic_consumes:
                remapped_field_name = inverted_non_generic_consumes[original_field_name]
            location = self.manifest.get_field_location(remapped_field_name)
            field_mapping[location].append(remapped_field_name)

        # Add the generic fields to the field mapping
        for field_name in self.generic_consumes:
            try:
                location = self.manifest.get_field_location(field_name)
            except KeyError:
                msg = f"The dataset field {field_name} is not present in the dataset field mapping."
                raise ValueError(msg)

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

        if self.consumes:
            dataframe = dataframe.rename(columns=self.consumes)

        return dataframe


class DaskDataWriter(DataIO):
    def __init__(
        self,
        *,
        manifest: Manifest,
        component_spec: ComponentSpec,
        produces: t.Optional[ProducesType] = None,
    ):
        super().__init__(manifest=manifest, component_spec=component_spec)
        self.produces = produces
        # fmt: off
        self.non_generic_produces, self.generic_produces = self._resolve_custom_produces()
        # fmt: on

    def _resolve_custom_produces(
        self,
    ) -> t.Tuple[t.Dict[str, str], t.Dict[str, pa.DataType]]:
        """
        Function that resolves the custom consumes to get generic and non-generic consumes fields.

        Returns:
            The generic and non-generic consumes fields.
        """
        generic_produces = {}
        non_generic_produces = {}

        if self.produces is not None:
            for dataset_field, component_field_or_type in self.produces.items():
                if isinstance(component_field_or_type, str):
                    non_generic_produces[dataset_field] = component_field_or_type
                elif isinstance(component_field_or_type, pa.DataType):
                    if self.component_spec.is_produces_generic:
                        generic_produces[dataset_field] = component_field_or_type
                    else:
                        msg = (
                            f"The component spec is not generic and the dataset field"
                            f" {dataset_field} is not defined in the component spec."
                        )
                        raise ValueError(msg)
                else:
                    msg = (
                        "The produces argument must be a dictionary with column names as keys and"
                        " mapping names or pyarrow data types as values."
                    )
                    raise ValueError(
                        msg,
                    )

        return non_generic_produces, generic_produces

    def get_component_produces(
        self,
        inverted_non_generic_produces: t.Optional[t.Dict[str, str]] = None,
    ) -> t.Dict[str, Field]:
        """Returns the columns to produce."""
        produces = {}

        # Remap the field name if it is a non-generic field and is defined in the custom
        for field_name, field in self.component_spec.produces.items():
            if (
                inverted_non_generic_produces
                and field_name in inverted_non_generic_produces
            ):
                mapped_field_name = inverted_non_generic_produces[field_name]
                field.name = mapped_field_name
                produces[mapped_field_name] = field
            else:
                produces[field_name] = field

        # Add the generic fields to the field mapping
        if self.generic_produces is not None:
            for field_name, field_type in self.generic_produces.items():
                produces[field_name] = Field(name=field_name, type=Type(field_type))

        return produces

    def write_dataframe(
        self,
        dataframe: dd.DataFrame,
        dask_client: t.Optional[Client] = None,
    ) -> None:
        inverted_non_generic_produces = {
            v: k for k, v in self.non_generic_produces.items()
        }
        component_produces = self.get_component_produces(inverted_non_generic_produces)

        columns_to_produce = [
            column_name for column_name, field in component_produces.items()
        ]

        dataframe.index = dataframe.index.rename(DEFAULT_INDEX_NAME)

        if inverted_non_generic_produces is not None:
            dataframe = dataframe.rename(columns=inverted_non_generic_produces)

        # validation that all columns are in the dataframe
        self.validate_dataframe_columns(dataframe, columns_to_produce)

        dataframe = dataframe[columns_to_produce]
        write_task = self._write_dataframe(dataframe, component_produces)

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

    def _write_dataframe(
        self,
        dataframe: dd.DataFrame,
        component_produces: t.Dict[str, Field],
    ) -> dd.core.Scalar:
        """Create dataframe writing task."""
        location = (
            f"{self.manifest.base_path}/{self.manifest.pipeline_name}/"
            f"{self.manifest.run_id}/{self.component_spec.component_folder_name}"
        )

        # Add the generic fields to the field mapping
        schema = {field.name: field.type.value for field in component_produces.values()}
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
