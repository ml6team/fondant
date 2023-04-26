"""This module defines the FondantDataset class, which is a wrapper around the manifest.
It also defines the FondantComponent class, which uses the FondantDataset class to manipulate data.
"""

import argparse
import json
import logging
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import dask.dataframe as dd

from fondant.component_spec import FondantComponentSpec, kubeflow2python_type, Argument
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

    def load_data(self, spec: FondantComponentSpec) -> dd.DataFrame:
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
        manifest_fields = [(field.name, Type[field.type]) for field in fields.values()]
        self.manifest.add_subset(name, fields=manifest_fields)

        # create expected schema
        expected_schema = {field.name: field.type for field in fields.values()}
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


class FondantComponent(ABC):
    """Abstract base class for a Fondant component"""

    def __init__(self):
        self.spec = FondantComponentSpec.from_file("fondant_component.yaml")

    @abstractmethod
    def _get_manifest(self, args: argparse.Namespace) -> Manifest:
        """Abstract method that returns the dataset manifest"""

    @abstractmethod
    def _add_and_parse_args(self) -> argparse.Namespace:
        """Abstract method that adds and parses the component arguments"""

    def get_component_arguments(self) -> t.Mapping[str, Argument]:
        """
        Dictionary representation of the input and output arguments of a component
        Returns:
            Input and output arguments of the component.
        """
        component_arguments = {}
        kubeflow_component_spec = self.spec.kubeflow_specification
        component_arguments.update(kubeflow_component_spec.input_arguments)
        component_arguments.update(kubeflow_component_spec.output_arguments)
        return component_arguments

    @abstractmethod
    def run(self):
        """Abstract method for running component"""


class FondantLoaderComponent(FondantComponent):
    """Abstract base class for a Fondant loader component"""

    def _add_and_parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        component_arguments = self.get_component_arguments()

        for arg in component_arguments.values():
            # Input manifest is not required for loading component
            if arg.name == "input_manifest_path":
                input_required = False
            else:
                input_required = True

            parser.add_argument(
                f"--{arg.name}",
                type=kubeflow2python_type[arg.type],
                required=input_required,
                help=arg.description,
            )

        # add metadata
        parser.add_argument(
            "--metadata",
            type=str,
            required=True,
            help="The metadata associated with the pipeline run",
        )

        return parser.parse_args()

    def _get_manifest(self, args: argparse.Namespace) -> Manifest:
        metadata = json.loads(args.metadata)
        # TODO ideally get rid of args.metadata by including them in the storage args, getting
        #  run_id based on args.output_manifest_path
        manifest = Manifest.create(
            base_path=metadata["base_path"],
            run_id=metadata["run_id"],
            component_id=metadata["component_id"],
        )

        return manifest

    @abstractmethod
    def load(self, args: argparse.Namespace) -> dd.DataFrame:
        """Abstract method that loads the initial dataframe"""

    def run(self):
        """
        Runs the loading component.
        """
        # step 1: Add and parse arguments
        args = self._add_and_parse_args()

        # step 2: Get manifest
        manifest = self._get_manifest(args)

        # step 3: Create dataset
        dataset = FondantDataset(manifest)

        # step 4: Load the dataframe according to the custom function provided to the user
        df = self.load(args)

        # step 5: Add index and specified subsets and write them to remote storage
        dataset.add_index(df)
        dataset.add_subsets(df, self.spec)

        # step 6: create and upload the output manifest
        dataset.upload(save_path=args.output_manifest_path)


class FondantTransformComponent(FondantComponent):
    """Abstract base class for a Fondant transform component"""

    def _add_and_parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        component_arguments = self.get_component_arguments()

        for arg in component_arguments.values():
            parser.add_argument(
                f"--{arg.name}",
                type=kubeflow2python_type[arg.type],
                required=True,
                help=arg.description,
            )

        # add metadata
        parser.add_argument(
            "--metadata",
            type=str,
            required=True,
            help="The metadata associated with the pipeline run",
        )

        return parser.parse_args()

    def _get_manifest(self, args: argparse.Namespace) -> Manifest:
        return Manifest.from_file(args.input_manifest_path)

    @abstractmethod
    def transform(
        self, args: argparse.Namespace, dataframe: dd.DataFrame
    ) -> dd.DataFrame:
        """Abstract method for applying data transformations to the input dataframe"""

    def run(self):
        """
        Runs the loading component.
        """
        # step 1: Add and parse arguments
        args = self._add_and_parse_args()

        # step 2: Get manifest
        manifest = self._get_manifest(args)

        # step 3: Create dataset
        dataset = FondantDataset(manifest)

        # step 5: Load the dataframe according to the component specifications
        df = dataset.load_data(self.spec)

        # provide this datasets to the user
        df = self.transform(args=args, dataframe=df)
        # TODO update index, potentially add new subsets (functionality still missing)

        # step 6: create and upload the output manifest
        dataset.upload(save_path=args.output_manifest_path)
