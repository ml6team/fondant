import os
import tempfile
from abc import ABC, abstractmethod
from typing import List, Collection, Optional, Dict

import pyarrow as pa
import pyarrow.compute as pc
from google.cloud import storage
from pyarrow.dataset import Scanner

from .common import ExpressDatasetHandler, ExpressDataset, ExpressTransformComponent, \
    ExpressDatasetDraft, ExpressLoaderComponent
from express.helpers import storage_helpers, parquet_helpers
from express.helpers.manifest_helpers import DataManifest, DataSource, DataType

PyArrowDatasetDraft = ExpressDatasetDraft[List[str], Scanner]


class PyArrowDataset(ExpressDataset[List[str], Scanner]):
    def load_index(self) -> List[str]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # TODO rework so it works with folder locations and not just files
            local_parquet_path = storage_helpers.download_file_from_bucket(
                storage.Client(), self.manifest.index.location, tmp_dir)
            return parquet_helpers.get_column_list_from_parquet(local_parquet_path, column_name="index")

    def _load_data_source(self, data_source: DataSource, index_filter=Optional[Collection[str]]) \
            -> Scanner:
        if data_source.type != DataType.parquet:
            raise TypeError("Only reading from parquet is currently supported.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # TODO rework so it works with folder locations and not just files
            local_parquet_path = storage_helpers.download_file_from_bucket(
                storage.Client(), data_source.location, tmp_dir)
            filters = None
            if index_filter:
                filters = (pc.field("index").isin(index_filter))
            return parquet_helpers.load_parquet_file(local_parquet_path, filters=filters)


class PyArrowDatasetHandler(ExpressDatasetHandler[List[str], Scanner]):

    @classmethod
    def _upload_index(cls, index: List[str], remote_path: str) -> DataSource:
        with tempfile.NamedTemporaryFile() as temp_parquet_file:
            # TODO: uploading without writing to temp file
            # TODO: sharded parquet
            parquet_helpers.write_index_parquet(
                index_parquet_path=temp_parquet_file.name,
                data_iterable_producer=lambda: index
            )
            bucket_name, _, blob_path = remote_path.partition("gs://")[2].partition("/")
            blob_file_path = f"{blob_path}/index.parquet"
            storage_helpers.upload_file_to_bucket(storage_client=storage.Client(),
                                                  file_to_upload_path=temp_parquet_file.name,
                                                  bucket_name=bucket_name,
                                                  blob_path=blob_file_path)
            return DataSource(
                location=remote_path,
                type=DataType.parquet,
                extensions=["parquet"],
                n_files=1,
                n_items=len(index)
            )

    @classmethod
    def _upload_data_source(cls, name: str, data: Scanner, remote_path: str) -> DataSource:
        with tempfile.TemporaryDirectory() as temp_folder:
            # TODO: can do streaming write and maybe keep track of num rows along the way.
            #   Can't use Scanner::count_rows because it's potentially read-once.
            pa.dataset.write_dataset(data, base_dir=temp_folder, format="parquet", partitioning=["index"])
            n_rows = parquet_helpers.get_nb_rows_from_parquet(temp_folder)
            to_upload = os.path.join(temp_folder, "files_to_upload.txt")
            with open(to_upload, "w") as out_file:
                for p in os.listdir(temp_folder):
                    if p.endswith(".parquet"):
                        out_file.write(os.path.join(temp_folder, p))
                        out_file.write("\n")
            storage_helpers.copy_files_bulk(to_upload, remote_path)
            return DataSource(
                location=remote_path,
                type=DataType.parquet,
                extensions=["parquet"],
                n_files=len(to_upload),
                n_items=n_rows
            )

    @classmethod
    def _load_dataset(cls, input_manifest: DataManifest) -> PyArrowDataset:
        return PyArrowDataset(input_manifest)


class PyArrowTransformComponent(PyArrowDatasetHandler, ExpressTransformComponent[List[str], Scanner], ABC):
    @classmethod
    @abstractmethod
    def transform(cls, data: PyArrowDataset, extra_args: Optional[Dict] = None) -> PyArrowDatasetDraft:
        pass


class PyArrowLoaderComponent(PyArrowDatasetHandler, ExpressLoaderComponent[List[str], Scanner], ABC):
    @classmethod
    @abstractmethod
    def load(cls, extra_args: Optional[Dict] = None) -> PyArrowDatasetDraft:
        pass
