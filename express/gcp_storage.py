"""
General helper class to handle gcp storage functionalities
"""
import subprocess  # nosec
import os
import logging
import tempfile
from typing import List
from urllib.parse import urlparse

from express.storage_interface import StorageHandlerInterface, DecodedBlobPath
from express import io

LOGGER = logging.getLogger(__name__)


class StorageHandler(StorageHandlerInterface):
    """Cloud storage handler class"""

    @staticmethod
    def decode_blob_path(fully_qualified_blob_path: str) -> DecodedBlobPath:
        """
        Function that decodes a given blob path to a list of [bucket_name, blob_path]
        Args:
             fully_qualified_blob_path (str): the fully qualified blob path that points to the blob
              containing the blob of interest
        Returns:
            DecodedBlobPath: a dataclass containing the bucket and blob path name
         """
        parsed_url = urlparse(fully_qualified_blob_path)
        if parsed_url.scheme == 'gs':
            bucket, blob_path = parsed_url.hostname, parsed_url.path[1:]
        else:
            path = parsed_url.path[1:].split('/', 1)
            bucket, blob_path = path[0], path[1]

        return DecodedBlobPath(bucket, blob_path)

    @staticmethod
    def construct_blob_path(bucket: str, blob_path: str) -> str:
        """
        Function that construct a fully qualified blob path from a bucket and a blob path
        Args:
            bucket (str): the bucket name
            blob_path (str): the blob path
        Returns
            str: the fully qualified blob path
         """

        return f"gs://{bucket}/{blob_path}"

    @staticmethod
    def get_blob_list(fully_qualified_blob_path: str) -> List[str]:
        """
        Function that returns the full list of blobs from a given path
        Args:
            fully_qualified_blob_path (str): the fully qualified blob path that points to the blob
            containing the blob of interest
        Returns:
            List [str]: the list of blobs
        """
        blob_list = subprocess.run(["gsutil", "ls", fully_qualified_blob_path],  # nosec
                                   capture_output=True, check=True).stdout.decode().split('\n')

        return blob_list

    @staticmethod
    def copy_folder(source: str, destination: str, copy_source_content: bool = False) -> str:
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
        if copy_source_content:
            if source[-1] != "\\":
                source = f"{source}\\*"
            else:
                source = f"{source}*"

        subprocess.run(  # nosec
            ["gsutil", '-o', '"GSUtil:use_gcloud_storage=True"', '-q', "-m", "cp", "-r",
             source, destination], check=True)

        LOGGER.info("Copying folder from %s to %s [DONE]", source, destination)

        folder_name = io.get_file_name(source)

        return os.path.join(destination, folder_name)

    @staticmethod
    def copy_files(source_files: List[str], destination: str):
        """
        Function that copies a source folder (or blob) from a remote/local source to a local/remote
        location respectively
        Args:
            source_files (List[str]): a list containing the url (local or remote) of the file to
             copy
            destination (str): the destination blob/folder to copy the files to
        """

        # Write file paths to a text file before piping
        with tempfile.TemporaryDirectory() as temp_folder:
            upload_text_file = os.path.join(temp_folder, "files_to_upload.txt")
            with open(upload_text_file, "w") as out_file:
                for file in source_files:
                    out_file.write(file)
                    out_file.write("\n")

            # Write files to tmp director
            pipe_file_list = subprocess.Popen(["cat", upload_text_file],  # nosec
                                              stdout=subprocess.PIPE)
            subprocess.call(  # nosec
                ['gsutil', '-o', '"GSUtil:use_gcloud_storage=True"', '-q', '-m', 'cp', '-I',
                 destination], stdin=pipe_file_list.stdout)

            LOGGER.info("A total of %s files were copied to %s", len(source_files), destination)

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
        self.copy_files([source_file], destination)
        file_name = io.get_file_name(source_file, return_extension=True)
        return os.path.join(destination, file_name)