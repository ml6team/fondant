"""Numeric analysis page of the app."""

import logging
import typing as t

import dask.dataframe as dd
import pandas as pd
import streamlit as st
from df_helpers.fields import get_numeric_fields
from interfaces.dataset_interface import DatasetLoaderApp
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

    @staticmethod
    def make_numeric_plot(
        dataframe: dd.DataFrame,
        numeric_field: t.List[str],
        plot_type: str,
    ):
        """Plots a numeric dataframe column with streamlit."""
        if plot_type == "histogram":
            data = dataframe[numeric_field].compute()  # .hist(bins=30)
            st.plotly_chart(data.hist(), use_container_width=True)

        elif plot_type == "violin":
            data = dataframe[numeric_field].compute()
            st.plotly_chart(data.plot(kind="violin"), use_container_width=True)

        elif plot_type == "density":
            data = dataframe[numeric_field].compute()
            st.plotly_chart(data.plot(kind="density_heatmap"), use_container_width=True)

        elif plot_type == "strip":
            data = dataframe[numeric_field].compute()
            st.plotly_chart(data.plot(kind="strip"), use_container_width=True)

        elif plot_type == "categorical":
            data = dataframe[numeric_field].value_counts()
            st.bar_chart(data.compute())

        else:
            msg = "Aggregation type not supported"
            raise ValueError(msg)

    def build_numeric_analysis_table(self, dataframe, numeric_fields) -> None:
        """Build the numeric analysis table."""
        # check if there are numeric fields
        if len(numeric_fields) == 0:
            st.warning("There are no numeric fields in this subset")
        else:
            st.write("## Numerical statistics")

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

    def build_numeric_analysis_plots(self, dataframe, numeric_fields):
        """Build the numeric analysis plots."""
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
                plot_type = st.selectbox(
                    "Plot type",
                    ["histogram", "violin", "density", "categorical"],
                )

            self.make_numeric_plot(dataframe, numeric_field, plot_type)

    def setup_app_page(self, dataframe, fields):
        numeric_fields = get_numeric_fields(fields)
        self.build_numeric_analysis_table(dataframe, numeric_fields)
        self.build_numeric_analysis_plots(dataframe, numeric_fields)


app = NumericAnalysisApp()
app.create_common_sidebar()
df, df_fields = app.create_loader_widget()
app.setup_app_page(df, df_fields)
