"""
This component crops images by removing the borders.
"""
import io
import logging
import typing as t

import numpy as np
from PIL import Image
import dask.dataframe as dd

from fondant.component import TransformComponent
from fondant.logger import configure_logging

from image_crop import remove_borders

configure_logging()
logger = logging.getLogger(__name__)


def extract_dimensions(image_df: dd.DataFrame) -> t.Tuple[np.int16, np.int16]:
    """Extract the width and height of an image

    Args:
        image_df (dd.DataFrame): input dataframe with images_data column

    Returns:
        np.int16: width of the image
        np.int16: height of the image
    """
    image = Image.open(io.BytesIO(image_df["images_data"]))

    return np.int16(image.size[0]), np.int16(image.size[1])


class ImageCroppingComponent(TransformComponent):
    """
    Component that crops images
    """

    def transform(
        self,
        *,
        dataframe: dd.DataFrame,
        cropping_threshold: int = -30,
        padding: int = 10
    ) -> dd.DataFrame:
        """
        Args:
            dataframe (dd.DataFrame): Dask dataframe
            cropping_threshold (int): threshold parameter used for detecting borders
            padding (int): padding for the image cropping

        Returns:
            dd.DataFrame: Dask dataframe with cropped images
        """
        # crop images
        dataframe["images_data"] = dataframe["images_data"].map(
            lambda x: remove_borders(x, cropping_threshold, padding),
            meta=("images_data", "bytes"),
        )

        # extract width and height
        dataframe[["images_width", "images_height"]] = dataframe[
            [
                "images_data",
            ]
        ].apply(extract_dimensions, axis=1, result_type="expand", meta={0: int, 1: int})

        return dataframe


if __name__ == "__main__":
    component = ImageCroppingComponent.from_file()
    component.run()
