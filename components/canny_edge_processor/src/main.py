"""
This component that segments images using a model from the Hugging Face hub.
"""
import io
import itertools
import logging
import toolz

import dask
import dask.dataframe as dd
from PIL import Image
import pandas as pd
import numpy as np
# from controlnet_aux import OpenposeDetector
# import torch
import cv2

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def transform(bytes: bytes,
              low_threshold: int = 100,
              high_threshold: int = 200) -> bytes:
    """transforms an image to a canny edge map

    Args:
        bytes (bytes): input image
        low_threshold (int, optional): low threshold for canny edge detection. Defaults to 100.
        high_threshold (int, optional): high threshold for canny edge detection. Defaults to 200.

    Returns:
        bytes: canny edge map
    """
    # load image
    image = Image.open(io.BytesIO(bytes)).convert("RGB")

    # convert to canny edge map
    image = cv2.Canny(np.array(image), low_threshold, high_threshold)[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # save image to bytes
    output_bytes = io.BytesIO()
    canny_image.save(output_bytes, format='JPEG')

    return output_bytes.getvalue()


class CannyEdgeComponent(TransformComponent):
    """
    Component that calculates the canny edge map of an image
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            low_threshold: low threshold for canny edge detection
            high_threshold: high threshold for canny edge detection

        Returns:
            Dask dataframe
        """

        dataframe['canny_data'] = dataframe['images_data'].apply(lambda x: transform(x, low_threshold, high_threshold),
                                                                 meta=('canny_data', 'bytes'))

        return dataframe


if __name__ == "__main__":
    component = CannyEdgeComponent.from_file()
    component.run()
