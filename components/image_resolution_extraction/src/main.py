"""This component filters images of the dataset based on image size (minimum height and width)."""
import io
import logging
import typing as t

import imagesize
import numpy as np
import pandas as pd

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


def extract_dimensions(images: bytes) -> t.Tuple[np.int16, np.int16]:
    """Extract the width and height of an image.

    Args:
        images: input dataframe with images_data column

    Returns:
        np.int16: width of the image
        np.int16: height of the image
    """
    width, height = imagesize.get(io.BytesIO(images))

    return np.int16(width), np.int16(height)


class ImageResolutionExtractionComponent(PandasTransformComponent):
    """Component that extracts image dimensions."""

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
        Returns:
            dataset.
        """
        logger.info("Filtering dataset...")

        dataframe[[("images", "width"), ("images", "height")]] = dataframe[
            [("images", "data")]
        ].apply(lambda x: extract_dimensions(x.images.data), axis=1)

        return dataframe


if __name__ == "__main__":
    component = ImageResolutionExtractionComponent.from_args()
    component.run()
