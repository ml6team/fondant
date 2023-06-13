from typing import List, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd
import streamlit as st
from data import load_manifest
from numeric_analysis import make_numeric_plot, make_numeric_statistics_table
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder
from table import configure_image_builder, convert_image_column

MANIFEST_PATH = (
    "/Users/bertchristiaens/Bitbucket/fondant_ml6/"
    "local_artifact/logo_manifest_embed_retrieval.json"
)


def build_sidebar() -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    Build the sidebar for the data explorer app.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[List[str]]]: Tuple with manifest path,
        subset name and fields
    """
    # text field for manifest path
    st.sidebar.title("Subset loader")
    manifest_path = st.sidebar.text_input("Manifest path", MANIFEST_PATH)

    # load manifest
    if not manifest_path:
        st.warning("Please provide a manifest path")
        manifest = None
    else:
        manifest = load_manifest(manifest_path)

    # choose subset
    if manifest:
        subsets = manifest.subsets.keys()
        subset = st.sidebar.selectbox("Subset", subsets)
    else:
        subset = None

    # filter on subset fields
    if subset:
        fields = manifest.subsets[subset].fields
        fields = st.sidebar.multiselect("Fields", fields, default=fields)
    else:
        fields = None

    return manifest_path, subset, fields


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
    rows = st.slider("Dataframe rows to load", 1, len(dataframe), 10)
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
        paginationPageSize=5, paginationAutoPageSize=False
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
    if len(numeric_fields) > 0:
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
    st.write("## Show numeric distributions")

    # choose a numeric field in dropdown
    numeric_field = st.selectbox("Field", numeric_fields)

    col1, col2 = st.columns(2)

    with col1:
        plot_type = st.selectbox("Plot type", ["bar", "line", "area"])
    with col2:
        aggregation_type = st.selectbox(
            "Aggregation type", ["histogram", "cumulative", "categorical"]
        )

    make_numeric_plot(dataframe, numeric_field, plot_type, aggregation_type)
