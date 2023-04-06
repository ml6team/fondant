"""Hugging Face Datasets single component module """

import os
import importlib
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union

from express.storage_interface import StorageHandlerModule
from express.manifest import DataManifest, DataSource
from express.import_utils import is_datasets_available
from .common import (
    ExpressDatasetHandler,
    ExpressDataset,
    ExpressTransformComponent,
    ExpressDatasetDraft,
    ExpressLoaderComponent,
)

if is_datasets_available():
    import datasets
    from datasets import load_dataset

# Define interface of pandas draft
# pylint: disable=unsubscriptable-object
HFDatasetsDatasetDraft = ExpressDatasetDraft[List[str], datasets.Dataset]

# pylint: disable=no-member
STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()


# pylint: disable=too-few-public-methods
class HFDatasetsDataset(ExpressDataset[List[str], datasets.Dataset]):
    """Hugging Face Datasets dataset"""

    def load_index(self) -> datasets.Dataset:
        """Function that loads in the index"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_parquet_path = STORAGE_HANDLER.copy_file(
                self.manifest.index.location, tmp_dir
            )

            # we specify "train" here to get a `Dataset` instead of a `DatasetDict`
            dataset = load_dataset(
                "parquet", data_files=local_parquet_path, split="train"
            )

            return dataset

    @staticmethod
    def _load_data_source(
        data_source: DataSource,
        index_filter: datasets.Dataset,
        **kwargs,
    ) -> datasets.Dataset:
        """Function that loads in a data source"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_source_location = data_source.location

            local_parquet_path = STORAGE_HANDLER.copy_parquet(
                data_source_location, tmp_dir
            )

            if "columns" in kwargs:
                if "index" not in kwargs["columns"]:
                    raise ValueError(
                        "Please also include the index when specifying columns"
                    )

            dataset = load_dataset(
                "parquet",
                data_files=local_parquet_path,
                split="train",
                **kwargs,
            )

            if index_filter:
                index = index_filter["index"]
                return dataset.filter(lambda example: example["index"] in index)

            return dataset


class HFDatasetsDatasetHandler(ExpressDatasetHandler[List[str], datasets.Dataset]):
    """Hugging Face Datasets Dataset handler"""

    @staticmethod
    def _upload_parquet(
        data: datasets.Dataset, name: str, remote_path: str
    ) -> DataSource:
        with tempfile.TemporaryDirectory() as temp_folder:
            # TODO: uploading without writing to temp file
            # TODO: sharded parquet? not sure if we should shard the index or only the data sources
            dataset_path = f"{temp_folder}/{name}.parquet"

            data.to_parquet(path_or_buf=dataset_path)

            fully_qualified_blob_path = f"{remote_path}/{name}.parquet"
            STORAGE_HANDLER.copy_file(
                source_file=dataset_path, destination=fully_qualified_blob_path
            )

            return DataSource(
                location=fully_qualified_blob_path,
                len=len(data),
                column_names=data.column_names,
            )

    @classmethod
    def _upload_index(cls, index: datasets.Dataset, remote_path: str) -> DataSource:
        data_source = cls._upload_parquet(
            data=index, name="index", remote_path=remote_path
        )
        return data_source

    @classmethod
    def _upload_data_source(
        cls,
        name: str,
        data: datasets.Dataset,
        remote_path: str,
    ) -> DataSource:
        data_source = cls._upload_parquet(data=data, name=name, remote_path=remote_path)
        return data_source

    @classmethod
    def _load_dataset(cls, input_manifest: DataManifest) -> HFDatasetsDataset:
        return HFDatasetsDataset(input_manifest)


class HFDatasetsTransformComponent(
    HFDatasetsDatasetHandler,
    ExpressTransformComponent[List[str], datasets.Dataset],
    ABC,
):
    """
    Hugging Face Datasets dataset transformer. Subclass this class to define custom
    transformation function
    """

    @classmethod
    @abstractmethod
    def transform(
        cls,
        data: HFDatasetsDataset,
        extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> HFDatasetsDatasetDraft:
        """Transform dataset"""


class HFDatasetsLoaderComponent(
    HFDatasetsDatasetHandler, ExpressLoaderComponent[List[str], datasets.Dataset], ABC
):
    """Hugging Face Datasets dataset loader. Subclass this class to define custom
    transformation function"""

    @classmethod
    @abstractmethod
    def load(
        cls, extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ) -> HFDatasetsDatasetDraft:
        """Load initial dataset"""
