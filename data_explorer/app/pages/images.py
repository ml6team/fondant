"""Image gallery page for the data explorer app."""
import io

import streamlit as st
from df_helpers.fields import get_image_fields
from interfaces.dataset_interface import DatasetLoaderApp
from PIL import Image


class ImageGalleryApp(DatasetLoaderApp):
    @staticmethod
    def setup_app_page(dataframe, fields):
        image_fields = get_image_fields(fields)

        if len(image_fields) == 0:
            st.warning("There are no image fields in this component")
        else:
            image_field = st.selectbox("Image field", image_fields)

            images = dataframe[image_field]
            images = [Image.open(io.BytesIO(x)).resize((256, 256)) for x in images]

            # show images in a gallery
            cols = st.columns(5)
            for i, image in enumerate(images):
                cols[i % 5].image(image, use_column_width=True)


app = ImageGalleryApp()
app.create_common_interface()
field_mapping, selected_fields = app.get_fields_mapping()
dask_df = app.load_dask_dataframe(field_mapping)
df = app.load_pandas_dataframe(dask_df, field_mapping)
app.setup_app_page(df, selected_fields)
