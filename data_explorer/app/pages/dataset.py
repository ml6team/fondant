"""Data exploration page of the app."""

import streamlit as st
from df_helpers.fields import get_image_fields
from df_helpers.image_render import configure_image_builder, convert_image_column
from interfaces.dataset_interface import DatasetLoaderApp
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder


class DatasetExplorerApp(DatasetLoaderApp):
    @staticmethod
    def setup_app_page(dataframe, fields):
        """Build the dataframe explorer table."""
        image_fields = get_image_fields(fields)

        # get the first rows of the dataframe
        cols = st.columns(2)
        with cols[0]:
            rows = st.slider(
                "Dataframe rows to load",
                1,
                len(dataframe),
                min(len(dataframe), 20),
            )
        with cols[1]:
            rows_per_page = st.slider("Amount of rows per page", 5, 50, 10)

        dataframe_explorer = dataframe.head(rows).reset_index(drop=False)

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

        # configure pagination and sidebar
        options_builder.configure_pagination(
            paginationPageSize=rows_per_page,
            paginationAutoPageSize=False,
        )
        options_builder.configure_side_bar()

        # display the Ag Grid table
        AgGrid(
            dataframe_explorer,
            gridOptions=options_builder.build(),
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        )


app = DatasetExplorerApp()
app.create_common_interface()
df, df_fields = app.create_loader_widget()
app.setup_app_page(df, df_fields)
