"""This file contains data loading logic"""
import json
import logging
from typing import List
from urllib.parse import urlparse

import dask.dataframe as dd
import streamlit as st
from exceptions import RemoteFileNotFoundException
from fsspec import open as fs_open

from fondant.manifest import Manifest

LOGGER = logging.getLogger(__name__)


def is_remote(path: str) -> bool:
    """Check if path is remote

    Args:
        path (str): path to check

    Returns:
        bool: whether path is remote
    """
    if urlparse(path).scheme in ['', 'file']:
        return False
    return True


@st.cache_data
def load_manifest(path: str) -> Manifest:
    """Load manifest from file path

    Args:
        path (str): file path

    Returns:
        Manifest: loaded manifest
    """
    remote_path = is_remote(path)
    # open the path and load the manifest
    try:
        with fs_open(path, encoding="utf-8") as file_:
            specification = json.load(file_)
    except Exception:
        if remote_path:
            raise RemoteFileNotFoundException("")
        else:
            raise FileNotFoundError(f"File {path} not found, please check if "
                                    "you have mounted the correct data directory"
                                    )
    try:
        manifest = Manifest(specification=specification)
    except Exception as exc:
        raise ValueError(f"File {path} is not a valid manifest file") from exc

    return manifest


@st.cache_data
def load_dataframe(manifest_path: str,
                   subset_name: str,
                   fields: List[str]) -> dd.DataFrame:
    """Load dataframe into dask dataframe

    Args:
        manifest_path (str): manifest file path
        subset_name (str): subset name to load
        fields (List[str]): column fields to load

    Returns:
        dd.DataFrame: dask dataframe with loaded data
    """
    manifest = load_manifest(manifest_path)

    subset = manifest.subsets[subset_name]
    subset_path = subset.location

    return dd.read_parquet(subset_path, columns=fields)
