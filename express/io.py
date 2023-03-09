"""
General I/O helpers function
"""
import pathlib
import logging
import os

logger = logging.getLogger(__name__)


def get_file_extension(file_name: str) -> str:
    """
    Function that returns a file extension from a file name
    Args:
        file_name (str): the file name to return the extension from
    Returns:
        (str): the file extension
    """
    return pathlib.Path(file_name).suffix[1:]


def get_file_name(file_uri: str, return_extension=False):
    """
    Function that returns the file name from a given gcs uri
    Args:
        file_uri (str): the file uri
        return_extension (bool): a boolean to indicate whether to return the file extension or not
    """
    path, extension = os.path.splitext(file_uri)
    file_name = os.path.basename(path)
    if return_extension:
        file_name = f"{file_name}{extension}"
    return file_name
