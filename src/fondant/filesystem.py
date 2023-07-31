"""This module defines common filesystem functionalities."""
import logging
import typing as t
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)

FSSPEC_SCHEME_DICT = {
    "file": "file",
    "s3": "s3",
    "gs": "gcs",
    "abfs": "abfs",
}


def get_filesystem(path_uri: str) -> fsspec.spec.AbstractFileSystem:
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

    if scheme in FSSPEC_SCHEME_DICT:
        return fsspec.filesystem(FSSPEC_SCHEME_DICT[scheme])

    msg = (
        f"Unable to create fsspec filesystem object for url `{path_uri}`"
        f" because of unsupported scheme: {scheme}.\nAvailable schemes "
        f"are {FSSPEC_SCHEME_DICT.keys()}"
    )

    raise ValueError(msg)


def list_files(path: t.Union[str, Path]) -> t.List[str]:
    """List files in the specified directory.

    Args:
        path: The path or URI of the directory.

    Returns:
        A list of absolute file paths in the specified directory.
    """
    fs = get_filesystem(str(path))
    return fs.ls(str(path))
