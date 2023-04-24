"""This module defines the FondantDataset class, which is a wrapper around the manifest.

It also defines the FondantComponent class, which uses the FondantDataset class to manipulate data.
"""

from abc import abstractmethod
import argparse
import json
from pathlib import Path
from typing import List, Mapping

import dask.dataframe as dd

from fondant.component_spec import ComponentSpec, kubeflow2python_type
from fondant.manifest import Manifest
from fondant.schema import Type, Field


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

    def _load_subset(self, name: str, fields: List[str]) -> dd.DataFrame:
        # get subset from the manifest
        subset = self.manifest.subsets[name]
        # TODO remove prefix
        location = "gcs://" + subset.location

        df = dd.read_parquet(
            location,
            columns=fields,
        )

        return df

    def load_data(self, spec: ComponentSpec) -> dd.DataFrame:
        subsets = []
        for name, subset in spec.input_subsets.items():
            fields = list(subset.fields.keys())
            subset_df = self._load_subset(name, fields)
            subsets.append(subset_df)

        # TODO this method should return a single dataframe with column_names called subset_field
        # TODO add index
        # df = concatenate_datasets(subsets)

        # return df

    def _upload_index(self, df: dd.DataFrame):
        # get location
        # TODO remove prefix and suffix
        remote_path = "gcs://" + self.manifest.index.location

        # upload to the cloud
        dd.to_parquet(
            df,
            remote_path,
            schema=self.index_schema,
            overwrite=True,
        )

    def _upload_subset(self, name: str, fields: Mapping[str, Field], df: dd.DataFrame):
        # add subset to the manifest
        manifest_fields = [(field.name, Type[field.type]) for field in fields.values()]
        self.manifest.add_subset(name, fields=manifest_fields)

        # create expected schema
        expected_schema = {field.name: field.type for field in fields.values()}
        expected_schema.update(self.index_schema)

        # TODO remove prefix
        remote_path = "gcs://" + self.manifest.subsets[name].location

        # upload to the cloud
        dd.to_parquet(df, remote_path, schema=expected_schema, overwrite=True)

    def add_index(self, df: dd.DataFrame):
        index_columns = list(self.manifest.index.fields.keys())

        # load index dataframe
        index_df = df[index_columns]

        self._upload_index(index_df)

    def add_subsets(self, df: dd.DataFrame, spec: ComponentSpec):
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


class FondantComponent:
    def __init__(self, type="transform"):
        # note: Fondant spec always needs to be called like this
        # and placed in the src directory
        self.spec = ComponentSpec.from_file("fondant_component.yaml")
        self.type = type

    def run(self) -> dd.DataFrame:
        """
        Parses input data, executes the transform, and creates the output manifest.

        Returns:
            Manifest: the output manifest
        """
        # step 1: add and parse arguments
        args = self._add_and_parse_args()
        # step 2: create Fondant dataset based on input manifest
        metadata = json.loads(args.metadata)
        if self.type == "load":
            # TODO ideally get rid of args.metadata
            # by including them in the storage args,
            # getting run_id based on args.output_manifest_path
            manifest = Manifest.create(
                base_path=metadata["base_path"],
                run_id=metadata["run_id"],
                component_id=metadata["component_id"],
            )
        else:
            manifest = Manifest.from_file(args.input_manifest_path)
        dataset = FondantDataset(manifest)
        # step 3: load or transform data
        if self.type == "load":
            df = self.load(args)
            dataset.add_index(df)
            dataset.add_subsets(df, self.spec)
        else:
            # create HF dataset, based on component spec
            input_dataset = dataset.load_data(self.spec)
            # provide this dataset to the user
            df = self.transform(
                dataset=input_dataset,
                args=args,
            )

        # step 4: create output manifest
        output_manifest = dataset.upload(save_path=args.output_manifest_path)

        return output_manifest

    def _add_and_parse_args(self):
        """
        Add and parse component arguments based on the component specification.
        """
        parser = argparse.ArgumentParser()

        kubeflow_component = self.spec.kubeflow_specification

        # add input args
        for arg in kubeflow_component.input_arguments.values():
            parser.add_argument(
                f"--{arg.name}",
                type=kubeflow2python_type[arg.type],
                required=False
                if self.type == "load" and arg.name == "input_manifest_path"
                else True,
                help=arg.description,
            )
        # add output args
        for arg in kubeflow_component.output_arguments.values():
            parser.add_argument(
                f"--{arg.name}",
                required=True,
                type=str,
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

    @abstractmethod
    def load(self, args) -> dd.DataFrame:
        """Load initial dataset"""

    @abstractmethod
    def transform(self, dataset, args) -> dd.DataFrame:
        """Transform existing dataset"""
