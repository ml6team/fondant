"""Main file of the data explorer interface"""
import logging
import argparse

import dask
import streamlit as st
from data import load_dataframe
from table import get_image_fields, get_numeric_fields
from widgets import (build_explorer_table, build_image_explorer,
                     build_numeric_analysis_plots,
                     build_numeric_analysis_table, build_sidebar)

dask.config.set({"dataframe.convert-string": False})

LOGGER = logging.getLogger(__name__)
# streamlit wide
st.set_page_config(layout="wide")


if __name__ == "__main__":
    # make sidebar with input fields for manifest path, subset and fields
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        help='Mounted or remote base path',
    )
    args = parser.parse_args()

    # TODO: add remote path as an argument (Later PR)
    manifest_path, subset, fields = build_sidebar()

    if fields and manifest_path and subset:
        # load dataframe
        dataframe_ = load_dataframe(manifest_path, subset, list(fields.keys()))

        # get partitions of dataframe
        if dataframe_.npartitions > 1:
            partitions = st.sidebar.slider("Partition", 1, dataframe_.npartitions, 0)
            dataframe = dataframe_.get_partition(partitions)
        else:
            dataframe = dataframe_

        # extract image and numeric columns
        image_fields = get_image_fields(fields)
        numeric_fields = get_numeric_fields(fields)
        # build tabs
        tab_explorer, tab_numeric, tab_images = st.tabs(
            ["Data explorer", "Numerical analysis", "Image explorer"]
        )

        # explorer tab with table of the dataset
        with tab_explorer:
            build_explorer_table(dataframe, image_fields)

        # numerical analysis tab with plots and statistics
        # of numerical columns
        with tab_numeric:
            build_numeric_analysis_table(dataframe, numeric_fields)
            build_numeric_analysis_plots(dataframe, numeric_fields)

        # tab for displaying image data
        with tab_images:
            build_image_explorer(dataframe, image_fields)
