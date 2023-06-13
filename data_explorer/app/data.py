"""This file contains data loading logic"""
from typing import List
import dask.dataframe as dd
import streamlit as st

from fondant.manifest import Manifest


@st.cache_data
def load_manifest(path: str) -> Manifest:
    """Load manifest from file path

    Args:
        path (str): file path

    Returns:
        Manifest: loaded manifest
    """
    manifest = Manifest.from_file(path)
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
