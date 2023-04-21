"""This module defines the FondantDataset class, which is a wrapper around the manifest.

It also defines the FondantComponent class, which uses the FondantDataset class to manipulate data.
"""

from abc import abstractmethod
import argparse
import json
from pathlib import Path

import dask.dataframe as dd

from express.component_spec import ExpressComponent, kubeflow2python_type
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

    def _load_subset(self, name: str) -> dd.DataFrame:
        # get subset from the manifest
        subset = self.manifest.get_subset(name)
        # get its location and fields
        # TODO remove gcp prefix
        location = "gcs://" + subset.location + ".parquet"
        fields = list(subset.fields.keys())

        df = dd.read_parquet(
            location, columns=fields, storage_options={"project": self.project_name}
        )

        return df

    def load_data(self, component_spec: ExpressComponent) -> dd.DataFrame:
        subsets = []
        for name in component_spec.input_subsets.keys():
            subset_data = self._load_subset(name)
            subsets.append(subset_data)

        # TODO this method should return a single dataframe with column_names called subset_field
        # TODO add index
        # dataset = concatenate_datasets(subsets)

        # return dataset

    def _upload_index(self, df) -> Index:
        # get location
        # TODO remove GCP prefix
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
        print("Fields:", fields)
        self.manifest.add_subset(name, fields=fields)
        # upload to the cloud
        # TODO remove prefix and suffix?
        remote_path = "gcs://" + self.manifest.get_subset(name).location + ".parquet"

        dd.to_parquet(
            df,
            remote_path,
            storage_options={"project": self.project_name},
            overwrite=True,
            schema={name: type_to_pyarrow[type_] for name, type_ in fields},
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
    type: str = "transform"

    @classmethod
    def _load_spec(cls) -> ExpressComponent:
        # note: Fondant spec always needs to be called like this
        # and placed in the src directory
        spec_path = "fondant_component.yaml"
        return ExpressComponent(spec_path)

    @classmethod
    def run(cls) -> dd.DataFrame:
        """
        Parses input data, executes the transform, and creates the output manifest.

        Returns:
            Manifest: the output manifest
        """
        # step 1: load component spec
        spec = cls._load_spec()
        # step 2: add and parse arguments
        args = cls._add_and_parse_args(spec)
        # step 3: create Fondant dataset based on input manifest
        metadata = json.loads(args.metadata)
        if cls.type == "load":
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
        if cls.type == "load":
            output_df = cls.load(args)
            dataset.add_index(output_df)
            dataset.add_subsets(output_df, spec)
        else:
            # create HF dataset, based on component spec
            input_dataset = dataset.load_data(spec)
            # provide this dataset to the user
            output_df = cls.transform(
                dataset=input_dataset,
                args=args,
            )

        # step 5: create output manifest
        output_manifest = dataset.upload(save_path=args.output_manifest_path)

        return output_manifest

    @classmethod
    def _add_and_parse_args(cls, spec):
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
                if cls.type == "load" and arg.name == "input_manifest_path"
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

    @classmethod
    @abstractmethod
    def load(cls, args) -> dd.DataFrame:
        """Load initial dataset"""

    @classmethod
    @abstractmethod
    def transform(cls, dataset, args) -> dd.DataFrame:
        """Transform existing dataset"""
