"""
This component loads dataset from files in a directory, these
files can be either in local directory or in remote location.
"""
import gzip
import logging
import os
import tarfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import dask.dataframe as dd
import fsspec
import pandas as pd
from dask import delayed
from fondant.component import LoadComponent

logger = logging.getLogger(__name__)


def decode_bytes(bytes):
    """
    Decodes bytes into str using the first successful encoding.

    This function tries to decode a byte sequence into a string
    using multiple encodings. The first successful decoding is returned.
    If none of the encodings are successful, None is returned.

    Parameters
    ----------
    bytes : bytes
        Bytes to be decoded.

    Returns:
    -------
    str or None
        Decoded string if successful, None otherwise.
    """
    encodings = ["utf-8", "latin-1", "iso-8859-1"]

    for enc in encodings:
        try:
            return bytes.decode(enc)
        except UnicodeDecodeError:
            logger.error("Failed to decode bytes !!!")
            continue

    return None


class FileHandler(ABC):
    """
    Abstract base class for file handlers.

    This class acts as a blueprint for all handler classes responsible
    for reading files from different formats and filesystems.

    Methods:
    -------
    read():
        Abstract method that must be implemented by subclasses.
    """

    def __init__(self, filepath, fs=None):
        """
        Initiate a new FileHandler with filepath and filesystem (fs).

        Parameters
        ----------
        filepath : str
            Path to the file to be read.
        fs : fsspec.AbstractFileSystem, optional
            Filesystem to use (default is local filesystem).
        """
        self.filepath = filepath
        self.fs = fs if fs else fsspec.filesystem("file")

    @abstractmethod
    def read(self):
        """Abstract method to read a file. Must be overridden by subclasses."""


class GzipFileHandler(FileHandler):
    """
    Handler for reading gzip compressed files.

    This class extends FileHandler to handle gzip compressed files.
    """

    def read(self):
        """
        Reads gzip compressed files and yields the contents of the file.

        Yields:
        ------
        str
            Content of the gzipped file.
        """
        logger.debug(f"Uncompressing {Path(self.filepath).name}......")
        with self.fs.open(self.filepath, "rb") as buffer, gzip.GzipFile(
            fileobj=buffer,
        ) as gz:
            yield self.filepath.split("/")[-1], decode_bytes(gz.read())


class ZipFileHandler(FileHandler):
    """
    Handler for reading zip compressed files.

    This class extends FileHandler to handle zip compressed files.
    """

    def read(self):
        """
        Reads zip compressed files and yields the content of each file in the archive.

        Yields:
        ------
        str
            Content of each file within the zipper archive.
        """
        logger.info(f"Uncompressing {Path(self.filepath).name}......")
        with self.fs.open(self.filepath, "rb") as buffer, zipfile.ZipFile(buffer) as z:
            for filename in z.namelist():
                print("filenmae: ", filename)
                with z.open(filename) as file_buffer:
                    yield filename.split("/")[-1], decode_bytes(file_buffer.read())


class TarFileHandler(FileHandler):
    """
    Handler for reading tar archived files.

    This class extends FileHandler to handle tar archived files.
    """

    def read(self):
        """
        Reads tar archived files and yields the content of each file in the archive.

        Yields:
        ------
        str
            Content of each file within the tar archive.
        """
        logger.info(f"Uncompressing {Path(self.filepath).name}......")
        with self.fs.open(self.filepath, "rb") as buffer, tarfile.open(
            fileobj=buffer,
        ) as tar:
            for tarinfo in tar:
                if tarinfo.isfile():
                    file = tar.extractfile(tarinfo)
                    yield tarinfo.name.split("/")[-1], decode_bytes(file.read())


class DirectoryHandler(FileHandler):
    """
    Handler for reading a directory of files.

    This class extends FileHandler to handle directories containing multiple files.
    """

    def read(self):
        """
        Reads a directory of files and yields the content of each file in the directory.

        Yields:
        ------
        str
            Content of each file within the directory.
        """
        logger.info(f"Loading files from {self.filepath} ......")
        filenames = self.fs.glob(os.path.join(self.filepath, "*"))
        for filename in filenames:
            with self.fs.open(filename, "r") as f:
                yield filename.split("/")[-1], decode_bytes(f.read())


def get_file_handler(filepath, fs):
    """
    This function returns an appropriate file handler based on the file extension
    of the input file.

    It supports .gz (gzip), .zip and .tar files. For any other file type, it defaults
    to a DirectoryHandler.

    Args:
    filepath (str): The file path (including name) to be processed. Should end with one
    of the supported extensions.
    fs (fsspec.spec.AbstractFileSystem): An instance of a FSSpec filesystem. This
    filesystem will be used to read the file.


    Returns:
    FileHandler: One of GzipFileHandler, ZipFileHandler, TarFileHandler or DirectoryHandler
      depending on the file extension.

    Raises:
    ValueError: If the file extension is not one of the supported ones (.gz, .zip, .tar).
    """
    if filepath.endswith(".gz"):
        return GzipFileHandler(filepath=filepath, fs=fs)
    if filepath.endswith(".zip"):
        return ZipFileHandler(filepath=filepath, fs=fs)
    if filepath.endswith(".tar"):
        return TarFileHandler(filepath=filepath, fs=fs)

    return DirectoryHandler(filepath)


class FilesToDaskConverter:
    """
    This class is responsible for converting file contents to a Dask DataFrame.

    Attributes:
    ----------
        handler : A user-provided handler that reads file content and name.
                  It is expected to be an object with a method named `read`
                  which should yield tuples containing (file_name, file_content).

    Methods:
    -------
        to_dask_dataframe():
            Converts the read file content to binary form and returns it as a Dask DataFrame.
    """

    def __init__(self, handler):
        """
        Constructs all the necessary attributes for the file converter object.

        Parameters
        ----------
            handler: handles the files and reads their content.
                     Should have a 'read' method that yields tuples (file_name, file_content).
        """
        self.handler = handler

    @staticmethod
    @delayed
    def create_record(file_name, file_content):
        """
        Static helper method that creates a dictionary record of file name and its content in
          string format.

        Parameters
        ----------
        file_name : str
            Name of the file.
        file_content : str
            The content of the file.

        Returns:
        -------
        dict
            A pandas dataframe with 'filename' as index and 'Content' column for binary file
            content.
        """
        record = pd.DataFrame(data={"filename": file_name, "Content": file_content})
        # convert 'Content' column to 'object' type if it's not already
        if record["Content"].dtype != "object":
            record["Content"] = record["Content"].astype("str")
        return record

    def to_dask_dataframe(self):
        """
        This method converts the read file content to binary form and returns a Dask DataFrame.

        Returns:
        ------
        dask.dataframe
            The created Dask DataFrame with filenames as indices and file content in binary form
            as Content column.
        """
        # Initialize an empty list to hold all our records in 'delayed' objects.
        # These objects encapsulate our function calls and their results, but don't execute
        # just yet.
        records = []
        logging.debug("Converting individual files into a dask dataframe ......")
        # Iterate over each file handled by the handler
        for file_name, file_content in self.handler.read():
            # Create a record for each file - note that this doesn't really run just yet due to
            # being 'delayed'
            record = self.create_record(file_name, file_content)
            # Add the 'delayed' record to our list
            records.append(record)

        #  Create an empty pandas dataframe with correct column names and types as meta.
        metadata = pd.DataFrame(
            data={
                "filename": pd.Series([], dtype="object"),
                "Content": pd.Series([], dtype="bytes"),
            },
        )

        # Use the delayed objects to create a Dask DataFrame. Dask knows how to handle these and
        # will efficiently compute them when needed.
        dataframe = dd.from_delayed(records, meta=metadata)

        # Set 'filename' as the index
        dataframe = dataframe.set_index("filename")

        return dataframe


class LoadFromFiles(LoadComponent):
    """Component that loads datasets from files."""

    def load(
        self,
        *,
        directory_path: str,
        fs: str,
    ) -> dd.DataFrame:
        """
        Args:
            directory_path: path to the directory with files. It can be either local
                            path or a remote path
            fs (str): Type of filesystem that will be used to read files, it can be
                      file/s3/az/gcp.

        Returns:
            Dataset: dataset with all file content in string format
        """
        fs = fsspec.filesystem(fs)
        # create a handler to read files from directory
        handler = get_file_handler(directory_path, fs=fs)

        # convert files to dask dataframe
        converter = FilesToDaskConverter(handler)
        return converter.to_dask_dataframe()


if __name__ == "__main__":
    component = LoadFromFiles.from_args()
    component.run()
