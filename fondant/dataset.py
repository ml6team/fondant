"""This module defines the FondantDataset class, which is a wrapper around the manifest.
It also defines the FondantComponent class, which uses the FondantDataset class to manipulate data.
"""

import logging
import typing as t
from pathlib import Path

import dask.dataframe as dd

from fondant.component_spec import FondantComponentSpec
from fondant.manifest import Manifest, Index
from fondant.schema import Type

logger = logging.getLogger(__name__)


class FondantDataset:
    """Wrapper around the manifest to download and upload data into a specific framework.

    Uses Dask Dataframes for the moment.
    """

    def __init__(self, manifest: Manifest):
        self.manifest = manifest
        self.mandatory_subset_columns = ["id", "source"]
        self.index_schema = {
            "source": "string",
            "id": "int64",
            "__null_dask_index__": "int64",
        }

    def _load_subset(
        self, subset_name: str, fields: t.List[str], index: t.Optional[Index] = None
    ) -> dd.DataFrame:
        """
        Function that loads the subset
        Args:
            subset_name: the name of the subset to load
            fields: the fields to load from the subset
            index: optional index to filter the subset on. If not provided, the default manifest
             index is used.
        Returns:
            The subset as a dask dataframe
        """

        subset = self.manifest.subsets[subset_name]
        remote_path = subset.location
        index_fields = list(self.manifest.index.fields.keys())
        fields = index_fields + fields

        logger.info(f"Loading subset {subset_name} with fields {fields}...")

        subset_df = dd.read_parquet(remote_path, columns=fields)

        # filter on default index of manifest if no index is provided
        if index is None:
            index_df = self._load_index()
            ids = index_df["id"].compute()
            sources = index_df["source"].compute()
            subset_df = subset_df[
                subset_df["id"].isin(ids) & subset_df["source"].isin(sources)
            ]

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
        Function that loads the index dataframe from the manifest
        Returns:
            The index as a dask dataframe
        """
        # get index subset from the manifest
        index = self.manifest.index
        # get remote path
        remote_path = index.location

        index_df = dd.read_parquet(remote_path)

        if list(index_df.columns) != ["id", "source"]:
            raise ValueError(
                f"Index columns should be 'id' and 'source', found {index_df.columns}"
            )

        return index_df

    def load_dataframe(self, spec: FondantComponentSpec) -> dd.DataFrame:
        """
        Function that loads the subsets defined in the component spec as a single dask dataframe
        Args:
            spec: the fondant component spec
        Returns:
            The dask dataframe with the field columns in the format (<subset>_<column_name>)
        """
        # load index into dataframe
        df = self._load_index()
        for name, subset in spec.input_subsets.items():
            fields = list(subset.fields.keys())
            subset_df = self._load_subset(name, fields)
            # left joins -> filter on index
            df = dd.merge(df, subset_df, on=["id", "source"], how="left")

        logging.info("Columns of dataframe:", list(df.columns))

        return df

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
        upload_index_task = dd.to_parquet(
            df, remote_path, schema=schema, overwrite=True, compute=False
        )
        return upload_index_task

    def upload_index(self, df: dd.DataFrame):
        """
        Create a Dask task that uploads the index dataframe to a remote location.
        Args:
            df: The input Dask dataframe.
        """
        remote_path = self.manifest.index.location
        index_columns = list(self.manifest.index.fields.keys())

        # load index dataframe
        index_df = df[index_columns]

        upload_index_task = self._create_write_dataframe_task(
            df=index_df, remote_path=remote_path, schema=self.index_schema
        )

        # Write index
        dd.compute(upload_index_task)

    def upload_subsets(self, df: dd.DataFrame, spec: FondantComponentSpec):
        """
        Create a list of Dask tasks for uploading the output subsets to remote locations.

        Args:
            df (dask.dataframe.DataFrame): The input Dask dataframe.
            spec (FondantComponentSpec): The specification of the output subsets.

        Raises:
            ValueError: If a field defined in an output subset is not present in the input
             dataframe.
        """

        def verify_subset_columns(subset_name, subset_fields, df):
            """
            Verify that all the fields defined in the output subset are present in
            the input dataframe
            """

            field_names = list(subset_fields.keys())
            subset_columns = [f"{subset_name}_{field}" for field in field_names]
            subset_columns.extend(self.mandatory_subset_columns)

            for col in subset_columns:
                if col not in df.columns:
                    raise ValueError(
                        f"Field {col} defined in output subset {subset_name} "
                        f"but not found in dataframe"
                    )

            return subset_columns

        def create_subset_dataframe(subset_name, subset_columns, df):
            """
            Create subset dataframe to save with the original field name as the column name
            """
            # Create a new dataframe with only the columns needed for the output subset
            subset_df = df[subset_columns]

            # Remove the subset prefix from the column names
            prefix_to_replace = f"{subset_name}_"
            subset_df = subset_df.rename(
                columns={
                    col: col[(len(prefix_to_replace)) :]
                    for col in subset_df.columns
                    if col not in self.mandatory_subset_columns
                    and col.startswith(prefix_to_replace)
                }
            )

            return subset_df

        upload_subsets_tasks = []

        # Loop through each output subset defined in the spec
        for subset_name, subset in spec.output_subsets.items():
            # Verify that all the fields defined in the output subset are present in the
            # input dataframe
            subset_columns = verify_subset_columns(subset_name, subset.fields, df)

            # Create a new dataframe with only the columns needed for the output subset
            subset_df = create_subset_dataframe(subset_name, subset_columns, df)

            # Get the remote path where the output subset should be uploaded
            remote_path = self.manifest.subsets[subset_name].location

            # Create the expected schema for the output subset
            expected_schema = {
                field.name: field.type.name for field in subset.fields.values()
            }
            expected_schema.update(self.index_schema)

            # Create a Dask task to upload the output subset to the remote location
            upload_subset_task = self._create_write_dataframe_task(
                df=subset_df, remote_path=remote_path, schema=expected_schema
            )
            upload_subsets_tasks.append(upload_subset_task)

        # Run all write subset tasks in parallel
        dd.compute(*upload_subsets_tasks)

    def upload_manifest(self, save_path: str):
        """
        Function that uploads the updated manifest to a remote storage
        Args:
            save_path: the path to upload the manifest to
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.manifest.to_file(save_path)
