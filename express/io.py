"""
General I/O helpers function
"""
import pathlib
import logging
import os
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def get_path_from_url(url) -> str:
    """
    Function that extract the path from a given url
    Args:
        url (str): the url to get the path from
    Returns:
        str: the url path
    """
    try:
        parsed_url = urlparse(url)
        path = f"{parsed_url.netloc}{parsed_url.path}"
    except Exception as e:
        raise Exception(e)
    return path


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


def create_subprocess_arguments(
        args: Optional[List[str]] = None, kwargs: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Function that creates subprocess arguments from a list of positional arguments
    and a dictionary of keyword arguments.

    Args:
        args (List[str]): A list of positional arguments to be included as subprocess arguments.
        kwargs (Dict[str, Any]): A dictionary of keyword arguments to be included as subprocess arguments.
    Returns:
        List[str]: A list of strings representing the subprocess arguments.
    """
    subprocess_args = []

    if args is not None:
        for arg in args:
            if "--" not in arg:
                subprocess_args.append(f"--{arg}")
            else:
                subprocess_args.append(arg)

    if kwargs is not None:
        for key, value in kwargs.items():
            key_prefix = "-" if key[0] != "-" else ""
            subprocess_args.append(f"{key_prefix}{key}")
            if isinstance(value, bool) or value is None:
                continue
            else:
                subprocess_args.append(str(value))

    return subprocess_args
