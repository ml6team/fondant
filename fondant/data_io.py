import logging
import typing as t

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from fondant.component_spec import ComponentSpec, ComponentSubset
from fondant.manifest import Manifest

logger = logging.getLogger(__name__)


class DataIO:
    def __init__(self, *, manifest: Manifest, component_spec: ComponentSpec) -> None:
        self.manifest = manifest
        self.component_spec = component_spec


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

        logger.info(f"Loading subset {subset_name} with fields {fields}...")

        subset_df = dd.read_parquet(remote_path, columns=fields)

        # add subset prefix to columns
        subset_df = subset_df.rename(
            columns={col: subset_name + "_" + col for col in subset_df.columns}
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
        index_df = dd.read_parquet(remote_path)

        return index_df

    def load_dataframe(self) -> dd.DataFrame:
        """
        Function that loads the subsets defined in the component spec as a single Dask dataframe for
          the user.

        Returns:
            The Dask dataframe with the field columns in the format (<subset>_<column_name>)
                as well as the index columns.
        """
        # load index into dataframe
        df = self._load_index()
        for name, subset in self.component_spec.consumes.items():
            fields = list(subset.fields.keys())
            subset_df = self._load_subset(name, fields)
            # left joins -> filter on index
            df = dd.merge(
                df,
                subset_df,
                left_index=True,
                right_index=True,
                how="left",
            )

        logging.info(f"Columns of dataframe: {list(df.columns)}")

        return df


class DaskDataWriter(DataIO):
    def write_dataframe(self, dataframe: dd.DataFrame) -> None:
        write_tasks = []

        dataframe.index = dataframe.index.rename("id").astype("string")

        # Turn index into an empty dataframe so we can write it
        index_df = dataframe.index.to_frame().drop(columns=["id"])
        write_index_task = self._write_subset(
            index_df, subset_name="index", subset_spec=self.component_spec.index
        )
        write_tasks.append(write_index_task)

        for subset_name, subset_spec in self.component_spec.produces.items():
            subset_df = self._extract_subset_dataframe(
                dataframe, subset_name=subset_name, subset_spec=subset_spec
            )
            write_subset_task = self._write_subset(
                subset_df, subset_name=subset_name, subset_spec=subset_spec
            )
            write_tasks.append(write_subset_task)

        with ProgressBar():
            logging.info("Writing data...")
            dd.compute(*write_tasks)

    @staticmethod
    def _extract_subset_dataframe(
        dataframe: dd.DataFrame, *, subset_name: str, subset_spec: ComponentSubset
    ) -> dd.DataFrame:
        """Create subset dataframe to save with the original field name as the column name."""
        # Create a new dataframe with only the columns needed for the output subset
        subset_columns = [f"{subset_name}_{field}" for field in subset_spec.fields]
        try:
            subset_df = dataframe[subset_columns]
        except KeyError as e:
            raise ValueError(
                f"Field {e.args[0]} defined in output subset {subset_name} "
                f"but not found in dataframe"
            )

        # Remove the subset prefix from the column names
        subset_df = subset_df.rename(
            columns={col: col[(len(f"{subset_name}_")) :] for col in subset_columns}
        )

        return subset_df

    def _write_subset(
        self, dataframe: dd.DataFrame, *, subset_name: str, subset_spec: ComponentSubset
    ) -> dd.core.Scalar:
        if subset_name == "index":
            location = self.manifest.index.location
        else:
            location = self.manifest.subsets[subset_name].location

        schema = {field.name: field.type.value for field in subset_spec.fields.values()}

        return self._create_write_task(dataframe, location=location, schema=schema)

    @staticmethod
    def _create_write_task(
        dataframe: dd.DataFrame, *, location: str, schema: t.Dict[str, str]
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
            dataframe, location, schema=schema, overwrite=False, compute=False
        )
        logging.info(f"Creating write task for: {location}")
        return write_task
