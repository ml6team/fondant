"""Pandas single component module """

import os
import importlib
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union

from express.components.common import (
    ExpressDatasetHandler,
    ExpressDataset,
    ExpressTransformComponent,
    ExpressDatasetDraft,
    ExpressLoaderComponent,
)
from express.manifest import DataManifest, DataSource, DataType
from express.storage_interface import StorageHandlerModule
from express.import_utils import is_pandas_available

if is_pandas_available():
    import pandas as pd

# Define interface of pandas draft
PandasDatasetDraft = ExpressDatasetDraft[List[str], Union[pd.DataFrame, pd.Series]]

STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()


class PandasDataset(ExpressDataset[List[str], Union[pd.DataFrame, pd.Series]]):
    """Pandas dataset"""

    def load_index(self) -> pd.Series:
        """Function that loads in the index"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_parquet_path = STORAGE_HANDLER.copy_file(
                self.manifest.index.location, tmp_dir
            )

            return pd.read_parquet(local_parquet_path).squeeze()

    @staticmethod
    def _load_data_source(
            data_source: DataSource,
            index_filter: Union[pd.DataFrame, pd.Series, List[str]],
            **kwargs,
    ) -> pd.DataFrame:
        if data_source.type != DataType.PARQUET:
            raise TypeError("Only reading from parquet is currently supported.")

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

            data_source_df = pd.read_parquet(local_parquet_path, **kwargs)

            if index_filter:
                return data_source_df.loc[index_filter]

            return data_source_df


class PandasDatasetHandler(ExpressDatasetHandler[List[str], pd.DataFrame]):
    """Pandas Dataset handler"""

    @staticmethod
    def _upload_parquet(
            data: Union[pd.DataFrame, pd.Series], name: str, remote_path: str
    ) -> DataSource:
        with tempfile.TemporaryDirectory() as temp_folder:
            # TODO: uploading without writing to temp file
            # TODO: sharded parquet? not sure if we should shard the index or only the data sources
            dataset_path = f"{temp_folder}/{name}.parquet"

            if isinstance(data, (pd.Index, pd.Series)):
                data = data.to_frame(name=name)

            data.to_parquet(path=dataset_path)

            fully_qualified_blob_path = f"{remote_path}/{name}.parquet"
            STORAGE_HANDLER.copy_file(
                source_file=dataset_path, destination=fully_qualified_blob_path
            )
            return DataSource(
                location=fully_qualified_blob_path,
                type=DataType.PARQUET,
                extensions=["parquet"],
                n_files=1,
                n_items=len(data),
            )

    @classmethod
    def _upload_index(
            cls, index: Union[pd.DataFrame, pd.Series, pd.Index], remote_path: str
    ) -> DataSource:
        data_source = cls._upload_parquet(
            data=index, name="index", remote_path=remote_path
        )
        return data_source

    @classmethod
    def _upload_data_source(
            cls, name: str, data: Union[pd.DataFrame, pd.Series, pd.Index], remote_path: str
    ) -> DataSource:
        data_source = cls._upload_parquet(data=data, name=name, remote_path=remote_path)
        return data_source

    @classmethod
    def _load_dataset(cls, input_manifest: DataManifest) -> PandasDataset:
        return PandasDataset(input_manifest)


class PandasTransformComponent(
    PandasDatasetHandler, ExpressTransformComponent[List[str], pd.DataFrame], ABC
):
    """Pandas dataset transformer. Subclass this class to define custom transformation function"""

    @classmethod
    @abstractmethod
    def transform(
            cls,
            data: PandasDataset,
            extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> PandasDatasetDraft:
        """Transform dataset"""


class PandasLoaderComponent(
    PandasDatasetHandler, ExpressLoaderComponent[List[str], pd.DataFrame], ABC
):
    """Pandas dataset loader. Subclass this class to define custom transformation function"""

    @classmethod
    @abstractmethod
    def load(
            cls, extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ) -> PandasDatasetDraft:
        """Load initial dataset"""
