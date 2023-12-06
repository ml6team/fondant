"""This component crops images by removing the borders."""
import io
import logging
import typing as t

import numpy as np
import pandas as pd
from fondant.component import PandasTransformComponent
from image_crop import remove_borders
from PIL import Image

logger = logging.getLogger(__name__)


def extract_dimensions(image_bytes: bytes) -> t.Tuple[np.int16, np.int16]:
    """Extract the width and height of an image.

    Args:
        image_bytes: input image as bytes

    Returns:
        np.int16: width of the image
        np.int16: height of the image
    """
    image = Image.open(io.BytesIO(image_bytes))

    return np.int16(image.size[0]), np.int16(image.size[1])


class ImageCroppingComponent(PandasTransformComponent):
    """Component that crops images."""

    def __init__(
        self,
        *,
        cropping_threshold: int,
        padding: int,
        **kwargs,
    ) -> None:
        """
        Args:
            cropping_threshold (int): threshold parameter used for detecting borders
            padding (int): padding for the image cropping.
            kwargs: Unhandled keyword arguments passed in by Fondant.
        """
        self.cropping_threshold = cropping_threshold
        self.padding = padding

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # crop images
        dataframe["image"] = dataframe["image"].apply(
            lambda image: remove_borders(image, self.cropping_threshold, self.padding),
        )

        # extract width and height
        dataframe["image_width", "image_height"] = dataframe["image"].apply(
            extract_dimensions,
            axis=1,
            result_type="expand",
            meta={0: int, 1: int},
        )

        return dataframe
