"""Main module for the data explorer interface."""

import argparse
from pathlib import Path

import dask
import fsspec
import streamlit as st
from config import SESSION_STATE_VARIABLES
from interfaces.common import render_common_interface
from interfaces.sidebar import render_sidebar
from pages.home import render_pipeline_overview
from st_pages import show_pages_from_config

dask.config.set({"dataframe.convert-string": False})


def initialize_state():
    for session_state_variable in SESSION_STATE_VARIABLES:
        if session_state_variable not in st.session_state:
            st.session_state[session_state_variable] = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        help="Mounted, remote, or local base path",
    )
    args, _ = parser.parse_known_args()

    st.session_state.base_path = Path(args.base_path)
    st.session_state.file_system, _ = fsspec.core.url_to_fs(args.base_path)


st.set_page_config(layout="wide")
initialize_state()

# Show streamlit page from pages.toml config file
show_pages_from_config()

render_sidebar()
render_common_interface()
render_pipeline_overview()
