import io
import logging

import pandas as pd
from fondant.component import PandasTransformComponent
from PIL import Image

logger = logging.getLogger(__name__)


class ResizeImagesComponent(PandasTransformComponent):
    """Component that resizes images based on a given width and height."""

    def __init__(self, *, resize_width: int, resize_height: int) -> None:
        self.resize_width = resize_width
        self.resize_height = resize_height

    def resize_image(self, img: bytes):
        image = Image.open(io.BytesIO(img)).convert("RGB")
        image = image.resize((self.resize_width, self.resize_height))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        return image_bytes.getvalue()

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info("Resizing images...")
        result = dataframe["image"].apply(
            self.resize_image,
            axis=1,
        )

        dataframe["image"] = result

        return dataframe
