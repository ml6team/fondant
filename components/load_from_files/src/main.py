"""
This component loads dataset from files in a directory, these
files can be either in local directory or in remote location.
"""
from __future__ import annotations

import gzip
import logging
import os
import tarfile
import zipfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Generator

import dask.dataframe as dd
import fsspec
import pandas as pd
from dask import delayed
from fondant.component import DaskLoadComponent

logger = logging.getLogger(__name__)


class AbstractFileHandler(ABC):
    """Abstract base class for file handlers."""

    def __init__(
            self,
            filepath: str,
            fs: fsspec.AbstractFileSystem | None = None,
    ) -> None:
        """
        Initiate a new AbstractFileHandler with filepath and filesystem (fs).

        Args:
          filepath : Path to the file to be read.
          fs : Filesystem to use (default is local filesystem).
        """
        self.filepath = filepath
        self.fs = fs if fs else fsspec.filesystem("file")

    @abstractmethod
    def read(self) -> Generator[tuple[str, BytesIO], None, None]:
        """Abstract method to read a file. Must be overridden by subclasses."""


class FileHandler(AbstractFileHandler):
    """Handler for reading files."""

    def read(self) -> Generator[tuple[str, BytesIO], None, None]:
        """
        Reads files and yields the contents of the file.

        Yields:
            Tuple consisting of filename and content of the file.
        """
        logger.debug(f"Reading file {self.filepath}....")
        with self.fs.open(self.filepath, "rb") as f:
            yield self.filepath.split("/")[-1], BytesIO(f.read())


class GzipFileHandler(AbstractFileHandler):
    """Handler for reading gzip compressed files."""

    def read(self) -> Generator[tuple[str, BytesIO], None, None]:
        """
        Reads gzip compressed files and yields the contents of the file.

        Yields:
            Tuple consisting of filename and content of the gzipped file.
        """
        logger.debug(f"Uncompressing {Path(self.filepath).name}......")
        with self.fs.open(self.filepath, "rb") as buffer, gzip.GzipFile(
                fileobj=buffer,
        ) as gz:
            yield self.filepath.split("/")[-1], BytesIO(gz.read())


class ZipFileHandler(AbstractFileHandler):
    """Handler for reading zip compressed files."""

    def read(self) -> Generator[tuple[str, BytesIO], None, None]:
        """
        Reads zip compressed files and yields the content of each file in the archive.

        Yields:
            Tuple consisting of filename and content of each file within the zipped archive.
        """
        logger.info(f"Uncompressing {Path(self.filepath).name}......")
        with self.fs.open(self.filepath, "rb") as buffer, zipfile.ZipFile(buffer) as z:
            for filename in z.namelist():
                with z.open(filename) as file_buffer:
                    buffer_content = file_buffer.read()
                    if not buffer_content:  # The buffer is empty.
                        continue
                    yield filename.split("/")[-1], BytesIO(buffer_content)


class TarFileHandler(AbstractFileHandler):
    """Handler for reading tar archived files."""

    def read(self) -> Generator[tuple[str, BytesIO], None, None]:
        """
        Reads tar archived files and yields the content of each file in the archive.

        Yields:
            Tuple consisting of filename and content of each file within the tar archive.
        """
        logger.info(f"Uncompressing {Path(self.filepath).name}......")
        with self.fs.open(self.filepath, "rb") as buffer, tarfile.open(
                fileobj=buffer,
        ) as tar:
            for tarinfo in tar:
                if tarinfo.isfile():
                    file = tar.extractfile(tarinfo)
                    if file is not None:
                        yield tarinfo.name.split("/")[-1], BytesIO(file.read())


class DirectoryHandler(AbstractFileHandler):
    """Handler for reading a directory of files."""

    def read(self) -> Generator[tuple[str, BytesIO], None, None]:
        """
        Reads a directory of files and yields the content of each file in the directory.

        Yields:
            Tuple consisting of filename and content of each file within the directory.
        """
        logger.info(f"Loading files from {self.filepath} ......")
        filenames = self.fs.glob(os.path.join(self.filepath, "*"))
        handler: AbstractFileHandler
        for filename in filenames:
            if filename.endswith(".gz"):
                handler = GzipFileHandler(filename, self.fs)
            elif filename.endswith(".zip"):
                handler = ZipFileHandler(filename, self.fs)
            elif filename.endswith(".tar"):
                handler = TarFileHandler(filename, self.fs)
            else:
                handler = FileHandler(filename, self.fs)
            yield from handler.read()


def get_file_handler(
        filepath: str,
        fs: fsspec.spec.AbstractFileSystem,
) -> AbstractFileHandler:
    """
    This function returns an appropriate file handler based on the file extension
    of the input file.

    It supports .gz (gzip), .zip and .tar files. For any other file type, it defaults
    to a DirectoryHandler.

    Args:
    filepath: The file path (including name) to be processed. Should end with one
    of the supported extensions.
    fs: An instance of a FSSpec filesystem. This
    filesystem will be used to read the file.


    Returns:
    AbstractFileHandler: One of GzipFileHandler, ZipFileHandler, TarFileHandler or DirectoryHandler
      depending on the file extension.

    Raises:
    ValueError: If the file extension is not one of the supported ones (.gz, .zip, .tar).
    """
    if filepath.endswith((".tar", ".tar.gz")):
        return TarFileHandler(filepath=filepath, fs=fs)
    if filepath.endswith(".gz"):
        return GzipFileHandler(filepath=filepath, fs=fs)
    if filepath.endswith(".zip"):
        return ZipFileHandler(filepath=filepath, fs=fs)

    return DirectoryHandler(filepath)


class FilesToDaskConverter:
    """This class is responsible for converting file contents to a Dask DataFrame."""

    def __init__(self, handler: AbstractFileHandler) -> None:
        """
        Constructs all the necessary attributes for the file converter object.

        Args:
            handler: Handles the files and reads their content.
                     Should have a 'read' method that yields tuples (file_name, file_content).
        """
        self.handler = handler

    @staticmethod
    @delayed
    def create_record(file_name: str, file_content: str) -> pd.DataFrame:
        """
        Static helper method that creates a dictionary record of file name and
        its content in string format.

        Args:
            file_name : Name of the file.
            file_content : The content of the file.

        Returns:
            A pandas dataframe with 'filename' as index and 'Content' column for
            binary file content.
        """
        if type(file_content) is tuple:
            file_content = file_content[0]
        return pd.DataFrame(
            data={"file_filename": [file_name], "file_content": [file_content]},
        )

    def to_dask_dataframe(self, chunksize: int = 1000) -> dd.DataFrame:
        """
        This method converts the read file content to binary form and returns a
        Dask DataFrame.

        Returns:
            The created Dask DataFrame with filenames as indices and file content
            in binary form as Content column.
        """
        # Initialize an empty list to hold all our records in 'delayed' objects.
        records = []
        temp_records = []

        # Iterate over each file handled by the handler
        for i, file_data in enumerate(self.handler.read()):
            file_name, file_content = file_data
            record = self.create_record(file_name, file_content)
            temp_records.append(record)

            if (i + 1) % chunksize == 0:
                # When we hit the chunk size, we combine all the records so far,
                # create a Delayed object, and add it to the list of partitions.
                records.extend(temp_records)
                temp_records = []

        # Take care of any remaining records
        if temp_records:
            records.extend(temp_records)

        # Create an empty pandas dataframe with correct column names and types as meta
        metadata = pd.DataFrame(
            data={
                "file_filename": pd.Series([], dtype="object"),
                "file_content": pd.Series([], dtype="bytes"),
            },
        )

        # Use the delayed objects to create a Dask DataFrame.
        return dd.from_delayed(records, meta=metadata)


def get_filesystem(path_uri: str) -> fsspec.spec.AbstractFileSystem | None:
    """Function to create fsspec.filesystem based on path_uri.

    Creates a abstract handle using fsspec.filesystem to
    remote or local directories to read files as if they
    are on device.

    Args:
        path_uri: can be either local or remote directory/fiel path

    Returns:
        A fsspec.filesystem (if path_uri is either local or belongs to
        one of these cloud sources s3, gcs or azure blob storage) or None
        if path_uri has invalid scheme
    """
    scheme = fsspec.utils.get_protocol(path_uri)

    if scheme == "file":
        return fsspec.filesystem("file")
    if scheme == "s3":
        return fsspec.filesystem("s3")
    if scheme == "gs":
        return fsspec.filesystem("gcs")
    if scheme == "abfs":
        return fsspec.filesystem("abfs")

    logger.warning(
        f"""Unable to create fsspec filesystem object
                    because of unsupported scheme: {scheme}""",
    )
    return None


class LoadFromFiles(DaskLoadComponent):
    """Component that loads datasets from files."""

    def __init__(self, *_, directory_uri: str) -> None:
        self.directory_uri = directory_uri

    def load(self) -> dd.DataFrame:
        """Loads dataset by reading all files in directory_uri."""
        fs = get_filesystem(self.directory_uri)
        if fs:
            # create a handler to read files from directory
            handler = get_file_handler(self.directory_uri, fs=fs)

            # convert files to dask dataframe
            converter = FilesToDaskConverter(handler)
            dataframe = converter.to_dask_dataframe()
            return dataframe
        logger.error(
            f"Could not load data from {self.directory_uri} because \
                     directory_uri doesn't belong to currently supported \
                     schemes: s3, gcs, abfs",
        )
        return None
