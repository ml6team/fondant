from abc import abstractmethod
import argparse
import importlib
import json
import tempfile
import os
from pathlib import Path

from datasets import Dataset, load_dataset

from express.component_spec import ExpressComponent
from express.manifest import Manifest
from express.storage_interface import StorageHandlerModule


STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()


class FondantDataset:
    """Wrapper around the manifest to download and upload data into a specific framework"""

    def __init__(self, manifest_path):
        self.manifest = Manifest.from_file(manifest_path)

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
        for subset_name in component_spec.input_subsets:
            subset = self.manifest.get_subset(subset_name)
            subset_data = self._load_subset(subset)
            data[subset_name] = subset_data

        # TODO this method should return a single dataframe with columnn_names called subset_field
        return data

    def upload(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(self.to_json_string(), encoding="utf-8")
        return None

    def to_json_string(self):
        return self.manifest.to_json()


class FondantComponent:
    type: str = "transform"

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
        # step 3: create or read input manifest
        dataset = FondantDataset(args.input_manifest)
        # step 4: transform data
        args = json.loads(args.args)
        if cls.type == "load":
            output_dataset = cls.load(args)
        else:
            # create HF dataset, based on component spec
            input_dataset = dataset.load_data(spec)
            # provide this dataset to the user
            output_dataset = cls.transform(dataset=input_dataset, args=args)

        # TODO check whether output dataset created by the user complies to spec.output_subsets

        # step 5: create output manifest based on component spec
        metadata = json.loads(args.metadata)
        output_manifest = dataset.upload(
            output_dataset, metadata=metadata, save_path=args.output_manifest
        )

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
            required=False,
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
