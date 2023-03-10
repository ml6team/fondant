"""
General interface class to unify storage functions across different cloud environments
"""

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class StorageHandlerModule:
    """Datclass to define module path for the different cloud Storage Handlers"""

    GCP: str = "express.gcp_storage"
    AWS: str = "express.aws_storage"
    AZURE: str = "express.azure_storage"


@dataclass_json
@dataclass
class DecodedBlobPath:
    """Dataclass for blob path construct"""

    bucket_name: str
    blob_path: str


class StorageHandlerInterface(ABC):
    """
    General helper class for a unified interface of storage helpers across different cloud
    platforms
    """

    @staticmethod
    @abstractmethod
    def decode_blob_path(fully_qualified_blob_path) -> DecodedBlobPath:
        """
        Function that decodes a given blob path to a list of
         [bucket_name, blob_path]
         Args:
         fully_qualified_blob_path (str): the fully qualified blob path that points to the blob
            containing the blob of interest
        """

    @staticmethod
    @abstractmethod
    def construct_blob_path(bucket: str, blob_path: str) -> str:
        """
        Function that construct a fully qualified blob path from a bucket and a blob path
        Args:
            bucket (str): the bucket name
            blob_path (str): the blob path
        Returns
            str: the fully qualified blob path
        """

    @staticmethod
    @abstractmethod
    def get_blob_list(fully_qualified_blob_path: str) -> List:
        """
        Function that returns the full list of blobs from a given path
        Args:
            fully_qualified_blob_path (str): the fully qualified blob path that points to the blob
            containing the blob of interest
        Returns:
            List: the list of blobs
        """

    @staticmethod
    @abstractmethod
    def copy_folder(
        source: str, destination: str, copy_source_content: bool = False
    ) -> str:
        """
        Function that copies a source folder (or blob) from a remote/local source to a local/remote
        location respectively
        Args:
            source (str): the source blob/folder
            destination (str): the destination blob/folder to copy the folder to
            copy_source_content (bool): whether to copy all the folder content of the source folder
              to the destination folder path (dump content of one folder to another)
        Returns
            str: the path to the destination copied folder
        """

    @abstractmethod
    def copy_file(self, source_file: str, destination: str) -> str:
        """
        Function that copies source files from a remote/local source to a local/remote
        location respectively
        Args:
             source_file (str): the url (local or remote) of the file to copy
            destination (str): the destination blob/folder to copy the files to
        Returns:
            str: the path where the file was copied to
        """

    @staticmethod
    @abstractmethod
    def copy_files(source_files: List[str], destination: str):
        """
        Function that copies source files from a remote/local source to a local/remote
        location respectively
        Args:
             source_files (Union[str, List[str]]): a list containing the urls (local or remote) of
              the file to copy
            destination (str): the destination blob/folder to copy the files to
        """

    def copy_parquet(self, parquet_path: str, destination: str) -> str:
        """
        Function that copies source files from a remote/local source to a local/remote
        location respectively
        Args:
            parquet_path (str): path to parquet. Can point to a single parquet file or folder for
             partitions
            destination (str): the destination blob/folder to copy the files to
        Returns:
            str: the path where the parquet file/folder was copied to
        """
        if ".parquet" in parquet_path:
            local_parquet_path = self.copy_file(parquet_path, destination)
        else:
            local_parquet_path = self.copy_folder(parquet_path, destination)

        return local_parquet_path
