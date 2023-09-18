"""This module defines common filesystem functionalities."""
import logging

import fsspec

logger = logging.getLogger(__name__)


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
    protocol = fsspec.utils.get_protocol(base_path)
    try:
        return fsspec.filesystem(protocol)
    except ValueError:
        raise
