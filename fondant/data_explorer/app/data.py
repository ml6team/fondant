import dask.dataframe as dd
import streamlit as st

from fondant.manifest import Manifest


@st.cache_data
def load_manifest(path):
    # check if gs:// path
    manifest = Manifest.from_file(path)
    return manifest


@st.cache_data
def load_dataframe(manifest_path, subset_name, fields):
    subset = load_manifest(manifest_path).subsets[subset_name]
    remote_path = subset.location
    remote_path = "gs://soy-audio-379412_datasets/data_explorer/part.0.parquet"
    subset_df = dd.read_parquet(remote_path, columns=fields)
    subset_df = subset_df.rename(
        columns={col: subset_name + "_" + col for col in subset_df.columns}
    )
    return subset_df
