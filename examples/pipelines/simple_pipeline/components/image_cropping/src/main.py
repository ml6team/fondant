"""
This component crops images by removing the borders.
"""
import io
import logging

import numpy as np
from PIL import Image
import dask.dataframe as dd

from fondant.component import TransformComponent
from fondant.logger import configure_logging

from image_crop import remove_borders

configure_logging()
logger = logging.getLogger(__name__)


def extract_width(image_bytes: bytes) -> np.in16:
    """Extract the width of an image

    Args:
        image_bytes (bytes): input image in bytes

    Returns:
        np.int16: width of the image
    """
    return np.int16(
        Image.open(io.BytesIO(image_bytes)).size[0]
    )


def extract_height(image_bytes: bytes) -> np.int16:
    """Extract the height of an image

    Args:
        image_bytes (bytes): input image in bytes

    Returns:
        np.int16: height of the image
    """
    return np.int16(
        Image.open(io.BytesIO(image_bytes)).size[1]
    )


class ImageCroppingComponent(TransformComponent):
    """
    Component that crops images
    """

    def transform(
        self, *, dataframe: dd.DataFrame, scale: float = 2.0, offset: int = -100, padding: int = 10
    ) -> dd.DataFrame:
        """
        Args:
            dataframe (dd.DataFrame): Dask dataframe
            scale (float): scale parameter used for detecting borders
            offset (int): scale parameter used for detecting borders
            padding (int): padding for the image cropping

        Returns:
            dd.DataFrame: Dask dataframe with cropped images
        """
        # crop images
        dataframe["images_crop_data"] = dataframe["images_data"].map(lambda x: remove_borders(x, scale, offset, padding),
                                                                     meta=("images_crop_data", "bytes"))

        # extract width and height
        dataframe["images_crop_width"] = dataframe["images_data"].map(extract_width, meta=("images_crop_width", int))
        dataframe["images_crop_height"] = dataframe["images_data"].map(extract_height, meta=("images_crop_height", int))
        return dataframe


if __name__ == "__main__":
    component = ImageCroppingComponent.from_file()
    component.run()
