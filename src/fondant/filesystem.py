"""This module defines common filesystem functionalities."""
import logging
import typing as t
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)


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


def list_files(path: t.Union[str, Path]) -> t.List[str]:
    """List files in the specified directory.

    Args:
        path: The path or URI of the directory.

    Returns:
        A list of absolute file paths in the specified directory.
    """
    fs: fsspec.filesystem = get_filesystem(str(path))
    return fs.ls(str(path))
