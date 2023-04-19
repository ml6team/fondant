from abc import abstractmethod
import argparse
import importlib
import json
import tempfile
import os
from pathlib import Path

from datasets import Dataset, load_dataset

from express.component_spec import ExpressComponent
from express.manifest import Manifest, Subset
from express.storage_interface import StorageHandlerModule


STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()


class FondantDataset:
    """Wrapper around the manifest to download and upload data into a specific framework"""

    def __init__(self, manifest):
        self.manifest = manifest

    def _load_subset(self, subset):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_parquet_path = STORAGE_HANDLER.copy_parquet(subset.location, tmp_dir)

        dataset = load_dataset(
            "parquet",
            data_files=local_parquet_path,
            split="train",
            column_names=subset.fields,
        )

        return dataset

    def load_data(self, component_spec):
        data = {}
        for name, subset in component_spec.input_subsets.items():
            subset_data = self._load_subset(subset)
            data[name] = subset_data

        # TODO this method should return a single dataframe with columnn_names called subset_field
        return data

    @staticmethod
    def _upload_parquet(data: Dataset, name: str, remote_path: str):
        with tempfile.TemporaryDirectory() as temp_folder:
            dataset_path = f"{temp_folder}/{name}.parquet"

            data.to_parquet(path_or_buf=dataset_path)

            fully_qualified_blob_path = f"{remote_path}.parquet"
            STORAGE_HANDLER.copy_file(
                source_file=dataset_path, destination=fully_qualified_blob_path
            )

    def _upload_subset(self, name, fields, data: Dataset) -> Subset:
        print("Fields:", fields)
        fields = [(field.name, field.type) for field in fields.values()]
        print("Fields:", fields)
        self.manifest.add_subset(name, fields=fields)
        remote_path = os.path.join(
            self.manifest.base_path, self.manifest.get_subset(name)["location"]
        )
        subset = self._upload_parquet(data=data, name=name, remote_path=remote_path)
        return subset

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
            # add to the manifest
            self._upload_subset(name, subset.fields, subset_dataset)

    def upload(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(self.manifest.to_json(), encoding="utf-8")
        return None


class FondantComponent:
    type: str = "transform"

    @classmethod
    def _load_spec(cls, spec_path) -> ExpressComponent:
        return ExpressComponent(spec_path)

    @classmethod
    def run(cls) -> Dataset:
        """
        Parses input data, executes the transform, and creates output artifacts.

        Returns:
            Manifest: the output manifest
        """
        # step 1: parse arguments
        args = cls._parse_args()
        # step 2: load component spec
        spec = cls._load_spec(args.spec_path)
        # step 3: create Fondant dataset based on input manifest
        metadata = json.loads(args.metadata)
        if cls.type == "load":
            manifest = Manifest.create(
                base_path=metadata["base_path"],
                run_id=metadata["run_id"],
                component_id=metadata["component_id"],
            )
        else:
            manifest = Manifest.from_file(args.input_manifest_path)
        dataset = FondantDataset(manifest)
        # step 4: transform data
        if cls.type == "load":
            output_dataset = cls.load(json.loads(args.args))
        else:
            # create HF dataset, based on component spec
            input_dataset = dataset.load_data(spec)
            # provide this dataset to the user
            output_dataset = cls.transform(
                dataset=input_dataset, args=json.loads(args.args)
            )

        # TODO check whether output dataset created by the user complies to spec.output_subsets
        print("Created columns:", output_dataset.column_names)
        print("Output subsets:", spec.output_subsets)
        dataset.add_subsets(output_dataset, spec)

        # step 5: create output manifest based on component spec
        output_manifest = dataset.upload(save_path=args.output_manifest_path)

        return output_manifest

    @classmethod
    def _parse_args(cls):
        """
        Parse component arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input-manifest-path",
            type=str,
            required=True,
            help="Path to the input manifest",
        )
        parser.add_argument(
            "--spec-path",
            type=str,
            required=True,
            help="Path to the Fondant component spec yaml file",
        )
        parser.add_argument(
            "--args",
            type=str,
            required=False,
            help="Custom arguments for the component, passed as a json dict string",
        )
        parser.add_argument(
            "--metadata",
            type=str,
            required=True,
            help="The metadata associated with the pipeline run",
        )
        parser.add_argument(
            "--output-manifest-path",
            type=str,
            required=True,
            help="Path to the output manifest",
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
