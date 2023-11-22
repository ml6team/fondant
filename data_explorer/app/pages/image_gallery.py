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
            for i, image in enumerate(images[image_slider[0] : image_slider[1]]):
                cols[i % 5].image(image, use_column_width=True)


app = ImageGalleryApp()
app.create_common_sidebar()
df, df_fields = app.create_loader_widget()
app.setup_app_page(df, df_fields)
