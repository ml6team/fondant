"""This module defines the FondantDataset class, which is a wrapper around the manifest.
It also defines the FondantComponent class, which uses the FondantDataset class to manipulate data.
"""

import logging
import typing as t
from pathlib import Path

import dask.dataframe as dd

from fondant.component_spec import FondantComponentSpec
from fondant.manifest import Manifest, Index
from fondant.schema import Type, Field

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
        self, name: str, fields: t.List[str], index: Index = None
    ) -> dd.DataFrame:
        # get subset from the manifest
        subset = self.manifest.subsets[name]
        # get remote path
        remote_path = subset.location

        # add index fields
        index_fields = list(self.manifest.index.fields.keys())
        fields = index_fields + fields

        logger.info(f"Loading subset {name} with fields {fields}...")

        df = dd.read_parquet(
            remote_path,
            columns=fields,
        )

        # filter on default index of manifest if no index is provided
        if index is None:
            index_df = self._load_index()
            ids = index_df["id"].compute()
            sources = index_df["source"].compute()
            df = df[df["id"].isin(ids) & df["source"].isin(sources)]

        # add subset prefix to columns
        df = df.rename(
            columns={
                col: name + "_" + col for col in df.columns if col not in index_fields
            }
        )

        return df

    def _load_index(self):
        # get index subset from the manifest
        index = self.manifest.index
        # get remote path
        remote_path = index.location

        df = dd.read_parquet(remote_path)

        if list(df.columns) != ["id", "source"]:
            raise ValueError(
                f"Index columns should be 'id' and 'source', found {df.columns}"
            )

        return df

    def load_dataframe(self, spec: FondantComponentSpec) -> dd.DataFrame:
        subset_dfs = []
        for name, subset in spec.input_subsets.items():
            fields = list(subset.fields.keys())
            subset_df = self._load_subset(name, fields)
            subset_dfs.append(subset_df)

        # return a single dataframe with column_names called subset_field
        # TODO perhaps leverage dd.merge here instead
        df = dd.concat(subset_dfs)

        logging.info("Columns of dataframe:", list(df.columns))

        return df

    def _upload_index(self, df: dd.DataFrame):
        # get remote path
        remote_path = self.manifest.index.location

        # upload to the cloud
        dd.to_parquet(
            df,
            remote_path,
            schema=self.index_schema,
            overwrite=True,
        )

    def _upload_subset(
        self, name: str, fields: t.Mapping[str, Field], df: dd.DataFrame
    ):
        # add subset to the manifest
        manifest_fields = [
            (field.name, Type[field.type.name]) for field in fields.values()
        ]
        self.manifest.add_subset(name, fields=manifest_fields)

        # create expected schema
        expected_schema = {field.name: field.type.name for field in fields.values()}
        expected_schema.update(self.index_schema)

        # get remote path
        remote_path = self.manifest.subsets[name].location
        # upload to the cloud
        dd.to_parquet(df, remote_path, schema=expected_schema, overwrite=True)

    def add_index(self, df: dd.DataFrame):
        index_columns = list(self.manifest.index.fields.keys())

        # load index dataframe
        index_df = df[index_columns]

        self._upload_index(index_df)

    def add_subsets(self, df: dd.DataFrame, spec: FondantComponentSpec):
        for name, subset in spec.output_subsets.items():
            fields = list(subset.fields.keys())
            # verify fields are present in the output dataframe
            subset_columns = [f"{name}_{field}" for field in fields]
            subset_columns.extend(self.mandatory_subset_columns)

            for col in subset_columns:
                if col not in df.columns:
                    raise ValueError(
                        f"Field {col} defined in output subset {name} but not found in dataframe"
                    )

            # load subset dataframe
            subset_df = df[subset_columns]
            # remove subset prefix from subset columns
            subset_df = subset_df.rename(
                columns={
                    col: col.split("_")[-1]
                    for col in subset_df.columns
                    if col not in self.mandatory_subset_columns
                }
            )
            # add to the manifest and upload
            self._upload_subset(name, subset.fields, subset_df)

    def upload(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.manifest.to_file(save_path)
        return None
