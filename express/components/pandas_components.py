"""Pandas single component module """
from pathlib import Path
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
from express.import_utils import is_pandas_available
from express.io import get_path_from_url

if is_pandas_available():
    import pandas as pd

# Define interface of pandas draft
PandasDatasetDraft = ExpressDatasetDraft[List[str], Union[pd.DataFrame, pd.Series]]


class PandasDataset(ExpressDataset[List[str], Union[pd.DataFrame, pd.Series]]):
    """Pandas dataset"""

    def load_index(self, mount_dir: str) -> pd.Series:
        """Function that loads in the index"""
        index_location = get_path_from_url(self.manifest.index.location)
        index_path = str(Path(mount_dir, index_location))

        return pd.read_parquet(index_path).squeeze()

    @staticmethod
    def _load_data_source(
            *,
            data_source: DataSource,
            mount_dir: str,
            index_filter: Union[pd.DataFrame, pd.Series, List[str]],
            **kwargs
    ) -> pd.DataFrame:
        if data_source.type != DataType.PARQUET:
            raise TypeError("Only reading from parquet is currently supported.")

        data_source_location = get_path_from_url(data_source.location)
        data_source_path = str(Path(mount_dir, data_source_location))
        data_source_df = pd.read_parquet(data_source_path)

        if index_filter:
            return data_source_df.loc[index_filter]

        return data_source_df


class PandasDatasetHandler(ExpressDatasetHandler[List[str], pd.DataFrame]):
    """Pandas Dataset handler"""

    @staticmethod
    def _upload_parquet(
            *,
            data: Union[pd.DataFrame, pd.Series],
            name: str,
            remote_path: str,
            mount_path: str,
    ) -> DataSource:
        # TODO: sharded parquet? not sure if we should shard the index or only the data sources
        Path(mount_path).mkdir(parents=True, exist_ok=True)

        # Convert to df to be able to write to parquet
        if isinstance(data, (pd.Index, pd.Series)):
            data = data.to_frame(name=name)

        data.to_parquet(path=mount_path)

        return DataSource(
            location=remote_path,
            type=DataType.PARQUET,
            extensions=["parquet"],
            n_files=1,
            n_items=len(data),
        )

    @classmethod
    def _upload_index(
            cls,
            index: Union[pd.DataFrame, pd.Series, pd.Index],
            remote_path: str,
            mount_path: str,
    ) -> DataSource:
        data_source = cls._upload_parquet(
            data=index, name="index", remote_path=remote_path, mount_path=mount_path
        )
        return data_source

    @classmethod
    def _upload_data_source(
            cls,
            *,
            name: str,
            data: Union[pd.DataFrame, pd.Series, pd.Index],
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
    ) -> PandasDataset:
        return PandasDataset(input_manifest, mount_dir)


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
