from abc import abstractmethod
import argparse
import importlib
import json
import tempfile
import os
from pathlib import Path

from datasets import Dataset, load_dataset, concatenate_datasets

from express.component_spec import ExpressComponent, kubeflow2python_type
from express.manifest import Manifest, Subset, Index
from express.storage_interface import StorageHandlerModule


STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()


class FondantDataset:
    """Wrapper around the manifest to download and upload data into a specific framework
    like HF datasets or Dask"""

    def __init__(self, manifest):
        self.manifest = manifest

    def _load_subset(self, name):
        # get subset from the manifest
        subset = self.manifest.get_subset(name)
        # get its location and fields
        location = subset.location
        fields = list(subset.fields.keys())

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_parquet_path = STORAGE_HANDLER.copy_parquet(location, tmp_dir)

        dataset = load_dataset(
            "parquet",
            data_files=local_parquet_path,
            split="train",
            column_names=fields,
        )

        return dataset

    def load_data(self, component_spec):
        subsets = []
        for name in component_spec.input_subsets.keys():
            subset_data = self._load_subset(name)
            subsets.append(subset_data)

        # TODO this method should return a single dataframe with column_names called subset_field
        # TODO add index
        dataset = concatenate_datasets(subsets)

        return dataset

    @staticmethod
    def _upload_parquet(data: Dataset, name: str, remote_path: str):
        with tempfile.TemporaryDirectory() as temp_folder:
            dataset_path = f"{temp_folder}/{name}.parquet"

            data.to_parquet(path_or_buf=dataset_path)

            fully_qualified_blob_path = f"{remote_path}.parquet"
            STORAGE_HANDLER.copy_file(
                source_file=dataset_path, destination=fully_qualified_blob_path
            )

    def _upload_index(self, data: Dataset) -> Index:
        # get location
        remote_path = os.path.join(
            self.manifest.base_path, "custom_artifact", self.manifest.index.location
        )
        print("Remote path for index:", remote_path)
        # upload to the cloud
        self._upload_parquet(data=data, name="index", remote_path=remote_path)

    def add_index(self, output_dataset):
        index_columns = list(self.manifest.index.fields.keys())
        # load subset data
        index_dataset = output_dataset.remove_columns(
            [col for col in output_dataset.column_names if col not in index_columns]
        )

        self._upload_index(index_dataset)

    def _upload_subset(self, name, fields, data: Dataset) -> Subset:
        # add subset to the manifest
        fields = [(field.name, field.type) for field in fields.values()]
        self.manifest.add_subset(name, fields=fields)
        # upload to the cloud
        remote_path = os.path.join(
            self.manifest.base_path, self.manifest.get_subset(name)["location"]
        )
        print(f"Remote path for subset {name}:", remote_path)
        self._upload_parquet(data=data, name=name, remote_path=remote_path)

    def add_subsets(self, output_dataset, component_spec):
        for name, subset in component_spec.output_subsets.items():
            fields = list(subset.fields.keys())
            # verify fields are present in the output dataset
            subset_columns = [f"{name}_{field}" for field in fields]
            for col in subset_columns:
                if col not in output_dataset.column_names:
                    raise ValueError(
                        f"Column {col} present in output subsets but not found in dataset"
                    )

            # load subset data
            subset_dataset = output_dataset.remove_columns(
                [
                    col
                    for col in output_dataset.column_names
                    if col not in subset_columns
                ]
            )
            # add to the manifest and upload
            self._upload_subset(name, subset.fields, subset_dataset)

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
    def run(cls) -> Dataset:
        """
        Parses input data, executes the transform, and creates output artifacts.

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
            manifest = Manifest.create(
                base_path=metadata[
                    "base_path"
                ],  # TODO make this part of the storage_args
                run_id=metadata[
                    "run_id"
                ],  # TODO get the run id based on args.output_manifest?
                component_id=metadata[
                    "component_id"
                ],  # TODO spec can be used to get component ID
            )
        else:
            manifest = Manifest.from_file(args.input_manifest_path)
        dataset = FondantDataset(manifest)
        # step 4: transform data
        if cls.type == "load":
            output_dataset = cls.load(args)
            dataset.add_index(output_dataset)
            dataset.add_subsets(output_dataset, spec)
        else:
            # create HF dataset, based on component spec
            input_dataset = dataset.load_data(spec)
            # provide this dataset to the user
            output_dataset = cls.transform(
                dataset=input_dataset,
                args=args,
            )

        # step 5: create output manifest
        output_manifest = dataset.upload(save_path=args.output_manifest_path)

        return output_manifest

    @classmethod
    def _add_and_parse_args(cls, spec):
        """
        Add component arguments
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
    def load(cls, args) -> Dataset:
        """Load initial dataset"""

    @classmethod
    @abstractmethod
    def transform(cls, dataset, args) -> Dataset:
        """Transform existing dataset"""
