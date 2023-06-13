import streamlit as st


def make_numeric_statistics_table(dataframe, fields):
    # make a new dataframe with statistics
    # (min, max, mean, std, median, mode)
    # for each numeric field
    statistics = []

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


def make_numeric_plot(dataframe, numeric_field, plot_type, aggregation_type):
    if aggregation_type == "histogram":
        data = dataframe[numeric_field].value_counts().compute()
    elif aggregation_type == "cumulative":
        data = dataframe[numeric_field].value_counts()
        data = data.cumsum().compute()
    elif aggregation_type == "categorical":
        data = dataframe[numeric_field].value_counts().compute()
        data = data.sort_index()
    else:
        raise ValueError("Aggregation type not supported")

    if plot_type == "bar":
        st.bar_chart(data)
    elif plot_type == "line":
        st.line_chart(data)
    elif plot_type == "area":
        st.area_chart(data)
