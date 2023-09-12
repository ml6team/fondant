"""Methods for building widget tabs and tables"""
import io
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from fondant.filesystem import get_filesystem
from fondant.manifest import Manifest

import dask.dataframe as dd
import pandas as pd
import streamlit as st
from numeric_analysis import make_numeric_plot, make_numeric_statistics_table
from PIL import Image
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder
from table import configure_image_builder, convert_image_column

LOGGER = logging.getLogger(__name__)


def build_sidebar(base_path: str) -> Tuple[Manifest, str, Dict[str, str]]:
    """
    Build the sidebar for the data explorer app.
    Args:
        base_path: the base path containing the pipeline runs
    Returns:
        Tuple[Optional[str], Optional[str], Optional[Dict]: Tuple with manifest path,
        subset name and fields
    """
    fs = get_filesystem(base_path)

    st.sidebar.title("Subset loader")
    st.sidebar.markdown(f"## Base path: \n {base_path}")
    st.sidebar.markdown(f"## Base path type: \n {fs.__class__.__name__}")

    # 1) List available pipelines
    available_pipelines = [os.path.basename(item) for item in fs.ls(base_path)]
    selected_pipeline = st.sidebar.selectbox("Select pipeline", available_pipelines)
    selected_pipeline_path = os.path.join(base_path, selected_pipeline)

    # 2) List available runs in descending order (most recent first)
    available_runs = [os.path.basename(item) for item in fs.ls(selected_pipeline_path)]
    available_runs.sort(reverse=True)
    selected_run = st.sidebar.selectbox("Select run", available_runs)
    selected_run_path = os.path.join(*[base_path, selected_pipeline, selected_run])

    # 3) List available components
    available_components = [os.path.basename(item) for item in fs.ls(selected_run_path)]
    selected_component = st.sidebar.selectbox("Select component", available_components)
    selected_component_path = os.path.join(
        *[base_path, selected_pipeline, selected_run, selected_component]
    )

    # 4) Find manifest
    # TODO: Currently, we search for a file with a manifest in it due to the old way that manifests
    #  are stored with a cache key. Disable later on to only search for `manifest.json`
    manifest_files = [os.path.basename(item) for item in fs.ls(selected_component_path) if
                      "manifest" in item]

    selected_manifest = None
    try:
        if len(manifest_files) > 1:
            st.warning(f"More than one manifest file found: {selected_manifest},"
                       f" selecting {selected_component_path}")
        selected_manifest = manifest_files[0]
    except IndexError:
        st.error(f"No manifest file was found in {selected_component_path}")

    manifest_path = os.path.join(
        *[base_path, selected_pipeline, selected_run, selected_component, selected_manifest]
    )
    manifest = Manifest.from_file(manifest_path)

    subsets = manifest.subsets.keys()
    subset = st.sidebar.selectbox("Subset", subsets)

    fields = manifest.subsets[subset].fields
    fields = st.sidebar.multiselect("Fields", fields, default=fields)
    field_types = {
        f"{field}": manifest.subsets[subset].fields[field].type.name for field in fields}

    return manifest, subset, field_types


def build_explorer_table(
        dataframe: Union[dd.DataFrame, pd.DataFrame], image_fields: List[str]
) -> None:
    """Build the dataframe explorer table.

    Args:
        dataframe (Union[dd.DataFrame, pd.DataFrame]): dataframe to explore
        image_fields (List[str]): list of image fields
    """
    st.write("## Dataframe explorer")
    st.write("In this table, you can explore the dataframe")

    # get the first rows of the dataframe
    cols = st.columns(2)
    with cols[0]:
        rows = st.slider("Dataframe rows to load", 1, len(dataframe), min(len(dataframe), 20))
    with cols[1]:
        rows_per_page = st.slider("Amount of rows per page", 5, 50, 10)

    dataframe_explorer = dataframe.head(rows)
    for field in image_fields:
        dataframe_explorer = convert_image_column(dataframe_explorer, field)

    # TODO: add formatting for other datatypes?

    # configure builder
    options_builder = GridOptionsBuilder.from_dataframe(dataframe_explorer)
    if len(image_fields) > 0:
        options_builder.configure_grid_options(rowHeight=100, rowWidth="auto")
    else:
        options_builder.configure_grid_options(rowHeight="auto", rowWidth="auto")

    # format the image columns
    for field in image_fields:
        configure_image_builder(options_builder, field)

    # configure pagination and side bar
    options_builder.configure_pagination(
        paginationPageSize=rows_per_page, paginationAutoPageSize=False
    )
    options_builder.configure_side_bar()

    # display the Ag Grid table
    AgGrid(
        dataframe_explorer,
        gridOptions=options_builder.build(),
        allow_unsafe_jscode=True,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
    )


def build_numeric_analysis_table(
        dataframe: Union[dd.DataFrame, pd.DataFrame], numeric_fields: List[str]
) -> None:
    """Build the numeric analysis table.

    Args:
        dataframe (Union[dd.DataFrame, pd.DataFrame]): dataframe to explore
        numeric_fields (List[str]): list of numeric fields
    """
    # check if there are numeric fields
    if len(numeric_fields) == 0:
        st.warning("There are no numeric fields in this subset")
    else:
        st.write("## Numerical statistics")

        # make numeric statistics table
        aggregation_dataframe = make_numeric_statistics_table(dataframe, numeric_fields)

        # configure the Ag Grid
        options_builder_statistics = GridOptionsBuilder.from_dataframe(
            aggregation_dataframe
        )
        options_builder_statistics.configure_grid_options(rowHeight="auto", rowWidth=10)

        # display the Ag Grid
        AgGrid(
            aggregation_dataframe,
            gridOptions=options_builder_statistics.build(),
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        )


def build_numeric_analysis_plots(
        dataframe: Union[dd.DataFrame, pd.DataFrame], numeric_fields: List[str]
) -> None:
    """Build the numeric analysis plots.

    Args:
        dataframe (Union[dd.DataFrame, pd.DataFrame]): dataframe to explore
        numeric_fields (List[str]): list of numeric fields
    """
    # check if there are numeric fields
    if len(numeric_fields) == 0:
        st.warning("There are no numeric fields in this subset")
    else:
        st.write("## Show numeric distributions")

        # choose a numeric field in dropdown
        cols = st.columns(2)
        with cols[0]:
            numeric_field = st.selectbox("Field", numeric_fields)
        with cols[1]:
            plot_type = st.selectbox("Plot type",
                                     ["histogram", "violin", "density", "categorical"])

        make_numeric_plot(dataframe, numeric_field, plot_type)


def build_image_explorer(dataframe: dd.DataFrame, image_fields: List[str]):
    """Build the image explorer
    This explorer shows a gallery of the images in a certain column.

    Args:
        dataframe (dd.DataFrame): dataframe to explore
        image_fields (List[str]): list of image fields
    """

    if len(image_fields) == 0:
        st.warning("There are no image fields in this subset")
    else:
        st.write("## Image explorer")
        st.write("In this table, you can explore the images")

        image_field = st.selectbox("Image field", image_fields)

        images = dataframe[image_field].compute()
        images = [Image.open(io.BytesIO(x)).resize((256, 256)) for x in images]

        image_slider = st.slider("image range", 0, len(images), (0, 10))

        # show images in a gallery
        cols = st.columns(5)
        for i, image in enumerate(images[image_slider[0]: image_slider[1]]):
            cols[i % 5].image(image, use_column_width=True)
