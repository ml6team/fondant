"""Hugging Face Datasets single component module """
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union
from pathlib import Path

from express.manifest import DataManifest, DataSource, DataType
from express.import_utils import is_datasets_available
from express.io import get_path_from_url
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


# pylint: disable=too-few-public-methods
class HFDatasetsDataset(ExpressDataset[List[str], datasets.Dataset]):
    """Hugging Face Datasets dataset"""

    def load_index(self, mount_dir: str) -> datasets.Dataset:
        """
        Function that loads in the index
        Args:
            mount_dir(str): the local directory mounted with FUSE
        """
        index_location = get_path_from_url(self.manifest.index.location)
        index_path = str(Path(mount_dir, index_location))

        return load_dataset("parquet", data_dir=index_path, split="train")

    @staticmethod
    def _load_data_source(
            *,
            data_source: DataSource,
            mount_dir: str,
            index_filter: datasets.Dataset,
            **kwargs,
    ) -> datasets.Dataset:
        """Function that loads in a data source"""
        if data_source.type != DataType.PARQUET:
            raise TypeError("Only reading from parquet is currently supported.")

        data_source_location = get_path_from_url(data_source.location)
        data_source_path = str(Path(mount_dir, data_source_location))

        if "columns" in kwargs:
            if "index" not in kwargs["columns"]:
                raise ValueError(
                    "Please also include the index when specifying columns"
                )

        dataset = load_dataset(
            "parquet", data_dir=data_source_path, split="train", **kwargs
        )

        if index_filter:
            index = index_filter["index"]
            return dataset.filter(lambda example: example["index"] in index)

        return dataset


class HFDatasetsDatasetHandler(ExpressDatasetHandler[List[str], datasets.Dataset]):
    """Hugging Face Datasets Dataset handler"""

    @staticmethod
    def _upload_parquet(
            *, data: datasets.Dataset, name: str, remote_path: str, mount_path: str
    ) -> DataSource:
        # TODO: sharded parquet? not sure if we should shard the index or only the data sources
        Path(mount_path).mkdir(parents=True, exist_ok=True)
        upload_path = f"{mount_path}/{name}.parquet"
        # TODO: Investigate why sharding is not possible with HF dataset (can only write to file)
        data.to_parquet(path_or_buf=upload_path)

        return DataSource(
            location=remote_path,
            type=DataType.PARQUET,
            extensions=["parquet"],
            n_files=1,
            n_items=len(data),
        )

    @classmethod
    def _upload_index(
            cls, *, index: datasets.Dataset, remote_path: str, mount_path: str
    ) -> DataSource:
        data_source = cls._upload_parquet(
            data=index, name="index", remote_path=remote_path, mount_path=mount_path
        )
        return data_source

    @classmethod
    def _upload_data_source(
            cls,
            name: str,
            data: datasets.Dataset,
            remote_path: str,
            mount_path: str,
    ) -> DataSource:
        data_source = cls._upload_parquet(
            data=data, name=name, remote_path=remote_path, mount_path=mount_path
        )
        return data_source

    @classmethod
    def _load_dataset(
            cls, input_manifest: DataManifest, mount_dir: str
    ) -> HFDatasetsDataset:
        return HFDatasetsDataset(input_manifest, mount_dir)


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
