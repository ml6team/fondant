"""Numeric analysis page of the app."""

import logging
import typing as t

import dask.dataframe as dd
import pandas as pd
import streamlit as st
from df_helpers.fields import get_numeric_fields
from interfaces.common import render_common_interface
from interfaces.dataset_interface import DatasetLoaderApp
from interfaces.sidebar import render_sidebar
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder

LOGGER = logging.getLogger(__name__)

pd.options.plotting.backend = "plotly"


class NumericAnalysisApp(DatasetLoaderApp):
    @staticmethod
    def make_numeric_statistics_table(
        dataframe: dd.DataFrame,
        numeric_fields: t.List[str],
    ) -> pd.DataFrame:
        """Make a table with statistics of numeric columns of the dataframe.

        Returns:
            pd.DataFrame: dataframe with column statistics
        """
        # make a new dataframe with statistics
        # for each numeric field
        dataframe[numeric_fields] = dataframe[numeric_fields].fillna(0)
        statistics = dataframe[numeric_fields].describe().compute()
        statistics = statistics.transpose()
        # add a column with the field name
        statistics["column_name"] = statistics.index
        # set this column as the first column
        column_order = [
            "column_name",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
        statistics = statistics[column_order]
        return statistics

    def build_numeric_analysis_table(
        self,
        dataframe: dd.DataFrame,
        numeric_fields: t.List[str],
    ) -> None:
        """Build the numeric analysis table."""
        # check if there are numeric fields
        if len(numeric_fields) == 0:
            st.warning("There are no numeric fields in this component")
        else:
            # make numeric statistics table
            aggregation_dataframe = self.make_numeric_statistics_table(
                dataframe,
                numeric_fields,
            )

            # configure the Ag Grid
            options_builder_statistics = GridOptionsBuilder.from_dataframe(
                aggregation_dataframe,
            )
            options_builder_statistics.configure_grid_options(
                rowHeight="auto",
                rowWidth=10,
            )

            # display the Ag Grid
            AgGrid(
                aggregation_dataframe,
                gridOptions=options_builder_statistics.build(),
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
            )

    def setup_app_page(self, dataframe: dd.DataFrame, fields):
        numeric_fields = get_numeric_fields(fields)
        self.build_numeric_analysis_table(dataframe, numeric_fields)


render_sidebar()
render_common_interface()

app = NumericAnalysisApp()
field_mapping, selected_fields = app.get_fields_mapping()
dask_df = app.load_dask_dataframe(field_mapping)
app.setup_app_page(dask_df, selected_fields)
