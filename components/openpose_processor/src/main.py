"""
This component that calculates HED maps from input images
"""
import io
import itertools
import logging
import toolz

import dask
import dask.dataframe as dd
from PIL import Image
import numpy as np
import torch
from controlnet_aux import OpenposeDetector

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def transform(bytes: bytes,
              model: OpenposeDetector) -> bytes:
    """transforms an image to a openpose map

    Args:
        bytes (bytes): input image
        model (OpenposeDetector): Openpose model

    Returns:
        bytes: openpose map
    """
    # load image
    image = Image.open(io.BytesIO(bytes)).convert("RGB")

    # calculate Openpose map
    hed_image = model(image, hand_and_face=True)

    # save image to bytes
    output_bytes = io.BytesIO()
    hed_image.save(output_bytes, format='JPEG')

    return output_bytes.getvalue()


class OpenposeProcessorComponent(TransformComponent):
    """
    Component that calculates the Openpose map of an image
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe

        Returns:
            Dask dataframe
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load Openpose model
        model = OpenposeDetector.from_pretrained('lllyasviel/ControlNet', device=device)

        # calculate HED map
        dataframe['openpose_data'] = dataframe['images_data'].apply(lambda x: transform(x, model), meta=('openpose_data', 'bytes'))

        return dataframe


if __name__ == "__main__":
    component = OpenposeProcessorComponent.from_file()
    component.run()
