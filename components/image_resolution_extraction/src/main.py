"""This component filters images of the dataset based on image size (minimum height and width)."""
import io
import logging
import typing as t

import dask.dataframe as dd
import imagesize
import numpy as np

from fondant.component import DaskTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def extract_dimensions(image_df: dd.DataFrame) -> t.Tuple[np.int16, np.int16]:
    """Extract the width and height of an image.

    Args:
        image_df (dd.DataFrame): input dataframe with images_data column

    Returns:
        np.int16: width of the image
        np.int16: height of the image
    """
    width, height = imagesize.get(io.BytesIO(image_df["images_data"]))

    return np.int16(width), np.int16(height)


class ImageResolutionExtractionComponent(DaskTransformComponent):
    """Component that extracts image dimensions."""

    def transform(self, *, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
        Returns:
            dataset.
        """
        logger.info("Length of the dataframe before filtering: %s", len(dataframe))

        logger.info("Filtering dataset...")

        dataframe[["images_width", "images_height"]] = \
            dataframe[["images_data"]].apply(extract_dimensions,
                                             axis=1, result_type="expand", meta={0: int, 1: int})

        return dataframe


if __name__ == "__main__":
    component = ImageResolutionExtractionComponent.from_args()
    component.run()
