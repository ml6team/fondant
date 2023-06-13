import io

import streamlit as st
from data import load_dataframe
from PIL import Image
from table import (
    get_image_fields,
    get_numeric_fields,
)
from widgets import (
    build_explorer_table,
    build_numeric_analysis_plots,
    build_numeric_analysis_table,
    build_sidebar,
)

# streamlit wide
st.set_page_config(layout="wide")


def build_image_explorer(dataframe, image_fields):
    """Build the image explorer
    This explorer shows a gallery of the images in a certain column.
    """
    st.write("## Image explorer")
    st.write("In this table, you can explore the images")

    image_field = st.selectbox("Image field", image_fields)

    images = dataframe[image_field].compute()
    images = [Image.open(io.BytesIO(x)).resize((256, 256)) for x in images]

    image_slider = st.slider("image range", 0, len(images), (0, 10))

    # show images in a gallery
    cols = st.columns(5)
    for i, image in enumerate(images[image_slider[0] : image_slider[1]]):
        cols[i % 5].image(image, use_column_width=True)


if __name__ == "__main__":
    # make sidebar with input fields for manifest path, subset and fields
    manifest_path, subset, fields = build_sidebar()

    if fields:
        # load dataframe
        dataframe_ = load_dataframe(manifest_path, subset, fields)
        dataframe_ = dataframe_.repartition(npartitions=3)

        # get partitions of dataframe
        if dataframe_.npartitions > 1:
            partitions = st.sidebar.slider("Partition", 1, dataframe_.npartitions, 0)
            dataframe = dataframe_.get_partition(partitions)
        else:
            dataframe = dataframe_

        # extract image and numeric columns
        image_fields = get_image_fields(dataframe)
        numeric_fields = get_numeric_fields(dataframe)

        # build tabs
        tab_explorer, tab_numeric, tab_images = st.tabs(
            ["Data explorer", "Numeric analysis", "Image explorer"]
        )

        with tab_explorer:
            build_explorer_table(dataframe, image_fields)
        with tab_numeric:
            build_numeric_analysis_table(dataframe, numeric_fields)
            build_numeric_analysis_plots(dataframe, numeric_fields)
        with tab_images:
            build_image_explorer(dataframe, image_fields)
