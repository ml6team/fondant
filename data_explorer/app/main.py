"""Main module for the data explorer interface."""

import logging

import dask
import streamlit as st
from interfaces.common_interface import MainSideBar
from st_pages import show_pages_from_config

LOGGER = logging.getLogger(__name__)

# streamlit wide
st.set_page_config(layout="wide")
dask.config.set({"dataframe.convert-string": False})


class PipelineOverviewApp(MainSideBar):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    app = PipelineOverviewApp()

    app.create_common_sidebar()

    # Show streamlit page from pages.toml config file
    show_pages_from_config()
