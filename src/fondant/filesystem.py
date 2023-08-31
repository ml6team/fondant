"""This module defines common filesystem functionalities."""
import logging
import typing as t
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)


def get_filesystem(
    reference_path: t.Union[str, Path],
) -> fsspec.spec.AbstractFileSystem:
    """Function to create fsspec.filesystem based on path_uri.
    Creates a abstract handle using fsspec.filesystem to
    remote or local directories to read files as if they
    are on device.

    Args:
        reference_path: The base path reference
    Returns:
        A fsspec.filesystem (if path_uri is either local or belongs to
        one of these cloud sources s3, gcs or azure blob storage) or None
        if path_uri has invalid scheme.
    """
    protocol = fsspec.utils.get_protocol(str(reference_path))
    try:
        fs = fsspec.filesystem(protocol)
        setattr(fs, "auto_mkdir", True)
        return fs
    except ValueError:
        raise
