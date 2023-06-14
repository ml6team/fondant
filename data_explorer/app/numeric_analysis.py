"""This file contains logic for numeric data analysis"""
import logging
from typing import List

import dask.dataframe as dd
import pandas as pd
import streamlit as st

LOGGER = logging.getLogger(__name__)

pd.options.plotting.backend = "plotly"


def make_numeric_statistics_table(dataframe: dd.DataFrame,
                                  fields: List[str]) -> pd.DataFrame:
    """Make a table with statistics of numeric columns of the dataframe

    Args:
        dataframe (dd.DataFrame): input dataframe
        fields (List[str]): numeric fields to calculate statistics on

    Returns:
        pd.DataFrame: dataframe with column statistics
    """
    # make a new dataframe with statistics
    # (min, max, mean, std, median, mode)
    # for each numeric field
    statistics = dataframe[fields].describe().compute()
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


def make_numeric_plot(dataframe: dd.DataFrame,
                      numeric_field: str,
                      plot_type: str):
    """Plots a numeric dataframe column with streamlit

    Args:
        dataframe (dd.DataFrame): input dataframe
        numeric_field (str): column name
        plot_type (str): type of plot

    Raises:
        ValueError: error if plot type is not in the options
    """
    if plot_type == "histogram":
        data = dataframe[numeric_field].compute()  # .hist(bins=30)
        st.plotly_chart(data.hist(), use_container_width=True)

    elif plot_type == "violin":
        data = dataframe[numeric_field].compute()
        st.plotly_chart(data.plot(kind='violin'), use_container_width=True)

    elif plot_type == "density":
        data = dataframe[numeric_field].compute()
        st.plotly_chart(data.plot(kind='density_heatmap'), use_container_width=True)

    elif plot_type == "strip":
        data = dataframe[numeric_field].compute()
        st.plotly_chart(data.plot(kind='strip'), use_container_width=True)

    elif plot_type == "categorical":
        data = dataframe[numeric_field].value_counts()
        st.bar_chart(data.compute())

    else:
        raise ValueError("Aggregation type not supported")
