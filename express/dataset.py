"""This module defines the FondantDataset class, which is a wrapper around the manifest.

It also defines the FondantComponent class, which uses the FondantDataset class to manipulate data.
"""

from abc import abstractmethod
import argparse
import json
from pathlib import Path

import dask.dataframe as dd

from express.component_spec import ComponentSpec, kubeflow2python_type
from express.manifest import Manifest, Subset, Index
from express.schema import type_to_pyarrow


class FondantDataset:
    """Wrapper around the manifest to download and upload data into a specific framework.

    Uses Dask Dataframes for the moment.
    """

    def __init__(self, manifest):
        self.manifest = manifest

    @property
    def project_name(self):
        return self.manifest.project_name

    def _load_subset(self, name: str, fields: list[str]) -> dd.DataFrame:
        # get subset from the manifest
        subset = self.manifest.subsets[name]
        # TODO remove prefix and suffix
        location = "gcs://" + subset.location + ".parquet"

        df = dd.read_parquet(
            location, columns=fields, storage_options={"project": self.project_name}
        )

        return df

    def load_data(self, component_spec: ComponentSpec) -> dd.DataFrame:
        subsets = []
        for name, subset in component_spec.input_subsets.items():
            fields = list(subset.fields.keys())
            subset_data = self._load_subset(name, fields)
            subsets.append(subset_data)

        # TODO this method should return a single dataframe with column_names called subset_field
        # TODO add index
        # df = concatenate_datasets(subsets)

        # return df

    def _upload_index(self, df) -> Index:
        # get location
        # TODO remove prefix and suffix
        remote_path = "gcs://" + self.manifest.index.location + ".parquet"
        # upload to the cloud
        dd.to_parquet(
            df,
            remote_path,
            storage_options={"project": self.project_name},
            overwrite=True,
        )

    def _upload_subset(self, name, fields, df) -> Subset:
        # add subset to the manifest
        fields = [(field.name, field.type) for field in fields.values()]
        self.manifest.add_subset(name, fields=fields)
        # upload to the cloud
        # TODO remove prefix and suffix?
        remote_path = "gcs://" + self.manifest.subsets[name].location + ".parquet"

        schema = {name: type_to_pyarrow[type_] for name, type_ in fields}

        print("Schema:", schema)

        dd.to_parquet(
            df,
            remote_path,
            storage_options={"project": self.project_name},
            overwrite=True,
            schema=schema,
        )

    def add_index(self, output_df):
        index_columns = list(self.manifest.index.fields.keys())
        # load index dataframe
        index_df = output_df[index_columns]

        self._upload_index(index_df)

    def add_subsets(self, output_df, component_spec):
        for name, subset in component_spec.output_subsets.items():
            fields = list(subset.fields.keys())
            # verify fields are present in the output dataframe
            subset_columns = [f"{name}_{field}" for field in fields]
            for col in subset_columns:
                if col not in output_df.columns:
                    raise ValueError(
                        f"Column {col} present in output subsets but not found in dataset"
                    )

            # load subset dataframe
            subset_df = output_df[subset_columns]
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
        self.spec = ComponentSpec("fondant_component.yaml")
        self.type = type

    def run(self) -> dd.DataFrame:
        """
        Parses input data, executes the transform, and creates the output manifest.

        Returns:
            Manifest: the output manifest
        """
        # step 1: add and parse arguments
        args = self._add_and_parse_args(self.spec)
        # step 3: create Fondant dataset based on input manifest
        metadata = json.loads(args.metadata)
        if self.type == "load":
            # TODO ideally get rid of arrs.metadata
            # by including them in the storage args,
            # getting run_id based on args.output_manifest_path
            manifest = Manifest.create(
                project_name=metadata["project_name"],
                base_path=metadata["base_path"],
                run_id=metadata["run_id"],
                component_id=metadata["component_id"],
            )
        else:
            manifest = Manifest.from_file(args.input_manifest_path)
        dataset = FondantDataset(manifest)
        # step 4: transform data
        if self.type == "load":
            output_df = self.load(args)
            dataset.add_index(output_df)
            dataset.add_subsets(output_df, self.spec)
        else:
            # create HF dataset, based on component spec
            input_dataset = dataset.load_data(self.spec)
            # provide this dataset to the user
            output_df = self.transform(
                dataset=input_dataset,
                args=args,
            )

        # step 5: create output manifest
        output_manifest = dataset.upload(save_path=args.output_manifest_path)

        return output_manifest

    def _add_and_parse_args(self, spec):
        """
        Add and parse component arguments based on the component specification.
        """
        parser = argparse.ArgumentParser()
        # add input args
        for arg in spec.input_arguments.values():
            parser.add_argument(
                f"--{arg.name}",
                type=kubeflow2python_type[arg.type],
                required=False
                if self.type == "load" and arg.name == "input_manifest_path"
                else True,
                help=arg.description,
            )
        # add output args
        for arg in spec.output_arguments.values():
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
