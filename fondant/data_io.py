import logging
import typing as t

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from fondant.component_spec import FondantComponentSpec
from fondant.manifest import Field, Manifest

logger = logging.getLogger(__name__)


class DataIO:
    def __init__(self, manifest: Manifest):
        self.manifest = manifest
        self.index_fields = ["id", "source"]
        self.index_schema = {
            "source": "string",
            "id": "int64",
            "__null_dask_index__": "int64",
        }


class DaskDataLoader(DataIO):
    def _load_subset(self, subset_name: str, fields: t.List[str]) -> dd.DataFrame:
        """
        Function that loads a subset from the manifest as a Dask dataframe.

        Args:
            subset_name: the name of the subset to load
            fields: the fields to load from the subset

        Returns:
            The subset as a dask dataframe
        """
        subset = self.manifest.subsets[subset_name]
        remote_path = subset.location
        index_fields = list(self.manifest.index.fields.keys())
        fields = index_fields + fields

        logger.info(f"Loading subset {subset_name} with fields {fields}...")

        subset_df = dd.read_parquet(remote_path, columns=fields)

        # add subset prefix to columns
        subset_df = subset_df.rename(
            columns={
                col: subset_name + "_" + col
                for col in subset_df.columns
                if col not in index_fields
            }
        )

        return subset_df

    def _load_index(self) -> dd.DataFrame:
        """
        Function that loads the index from the manifest as a Dask dataframe.

        Returns:
            The index as a dask dataframe
        """
        # get index subset from the manifest
        index = self.manifest.index
        # get remote path
        remote_path = index.location

        # load index from parquet, expecting id and source columns
        index_df = dd.read_parquet(remote_path, columns=["id", "source"])

        return index_df

    def load_dataframe(self, spec: FondantComponentSpec) -> dd.DataFrame:
        """
        Function that loads the subsets defined in the component spec as a single Dask dataframe for
          the user.

        Args:
            spec: the fondant component spec

        Returns:
            The Dask dataframe with the field columns in the format (<subset>_<column_name>)
                as well as the index columns.
        """
        # load index into dataframe
        df = self._load_index()
        for name, subset in spec.input_subsets.items():
            fields = list(subset.fields.keys())
            subset_df = self._load_subset(name, fields)
            # left joins -> filter on index
            df = dd.merge(df, subset_df, on=["id", "source"], how="left")

        logging.info(f"Columns of dataframe: {list(df.columns)}")

        return df


class DaskDataWriter(DataIO):
    @staticmethod
    def _create_write_dataframe_task(
        *, df: dd.DataFrame, remote_path: str, schema: t.Dict[str, str]
    ) -> dd.core.Scalar:
        """
        Creates a delayed Dask task to upload the given DataFrame to the remote storage location
         specified in the manifest.

        Args:
            df: The DataFrame to be uploaded.
            remote_path: the location to upload the subset to
            schema: the schema of the dataframe to write

        Returns:
             A delayed Dask task that uploads the DataFrame to the remote storage location when
              executed.
        """
        # Define task to upload index to remote storage
        write_task = dd.to_parquet(
            df, remote_path, schema=schema, overwrite=False, compute=False
        )
        logging.info(f"Creating write task for: {remote_path}")
        return write_task

    def write_index(self, df: dd.DataFrame):
        """
        Write the index dataframe to a remote location.

        Args:
            df: The output Dask dataframe returned by the user.
        """
        remote_path = self.manifest.index.location
        index_columns = list(self.manifest.index.fields.keys())

        # load index dataframe
        index_df = df[index_columns]

        # set index
        index_df.set_index("id")

        upload_index_task = self._create_write_dataframe_task(
            df=index_df, remote_path=remote_path, schema=self.index_schema
        )

        # Write index
        with ProgressBar():
            logging.info("Writing index...")
            dd.compute(upload_index_task)

    def write_subsets(self, df: dd.DataFrame, spec: FondantComponentSpec):
        """
        Write all subsets of the Dask dataframe to a remote location.

        Args:
            df (dask.dataframe.DataFrame): The output Dask dataframe returned by the user.
            spec (FondantComponentSpec): The component specification.

        Raises:
            ValueError: If a field defined in an output subset is not present in the user
             dataframe.
        """

        def verify_subset_columns(
            subset_name: str, subset_fields: t.Mapping[str, Field], df: dd.DataFrame
        ) -> t.List[str]:
            """
            Verify that all the fields defined in the output subset are present in
            the output dataframe.
            """
            # TODO: add logic for `additional fields`
            subset_columns = [f"{subset_name}_{field}" for field in subset_fields]
            subset_columns.extend(self.index_fields)
            for col in subset_columns:
                if col not in df.columns:
                    raise ValueError(
                        f"Field {col} defined in output subset {subset_name} "
                        f"but not found in dataframe"
                    )

            return subset_columns

        def create_subset_dataframe(
            subset_name: str, subset_columns: t.List[str], df: dd.DataFrame
        ):
            """Create subset dataframe to save with the original field name as the column name."""
            # Create a new dataframe with only the columns needed for the output subset
            subset_df = df[subset_columns]

            # Remove the subset prefix from the column names
            prefix_to_replace = f"{subset_name}_"
            subset_df = subset_df.rename(
                columns={
                    col: col[(len(prefix_to_replace)) :]
                    for col in subset_df.columns
                    if col not in self.index_fields
                    and col.startswith(prefix_to_replace)
                }
            )

            return subset_df

        upload_subsets_tasks = []

        # Loop through each output subset defined in the spec
        for subset_name, subset in spec.output_subsets.items():
            # Verify that all the fields defined in the output subset are present in the
            # output dataframe
            subset_columns = verify_subset_columns(subset_name, subset.fields, df)

            # Create a new dataframe with only the columns needed for the output subset
            subset_df = create_subset_dataframe(subset_name, subset_columns, df)

            # set index
            subset_df.set_index("id")

            # Get the remote path where the output subset should be uploaded
            remote_path = self.manifest.subsets[subset_name].location

            # Create the expected schema for the output subset
            expected_schema = {
                field.name: field.type.value for field in subset.fields.values()
            }
            expected_schema.update(self.index_schema)

            # Create a Dask task to upload the output subset to the remote location
            upload_subset_task = self._create_write_dataframe_task(
                df=subset_df, remote_path=remote_path, schema=expected_schema
            )
            upload_subsets_tasks.append(upload_subset_task)

        # Run all write subset tasks in parallel
        with ProgressBar():
            logging.info("Writing subsets...")
            dd.compute(*upload_subsets_tasks)
