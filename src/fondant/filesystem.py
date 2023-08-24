"""This module defines common filesystem functionalities."""
import logging

import fsspec

logger = logging.getLogger(__name__)

FSSPEC_SCHEME_DICT = {
    "file": "file",
    "s3": "s3",
    "gs": "gcs",
    "abfs": "abfs",
}


def get_filesystem(base_path: str) -> fsspec.spec.AbstractFileSystem:
    """Function to create fsspec.filesystem based on path_uri.
    Creates a abstract handle using fsspec.filesystem to
    remote or local directories to read files as if they
    are on device.

    Args:
        base_path: The base path reference
    Returns:
        A fsspec.filesystem (if path_uri is either local or belongs to
        one of these cloud sources s3, gcs or azure blob storage) or None
        if path_uri has invalid scheme.
    """
    scheme = fsspec.utils.get_protocol(base_path)

    if scheme in FSSPEC_SCHEME_DICT:
        return fsspec.filesystem(FSSPEC_SCHEME_DICT[scheme])

    msg = (
        f"Unable to create fsspec filesystem object for url `{base_path}`"
        f" because of unsupported scheme: {scheme}.\nAvailable schemes "
        f"are {list(FSSPEC_SCHEME_DICT.keys())}"
    )

    raise ValueError(msg)
