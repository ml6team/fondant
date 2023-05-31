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
from controlnet_aux import HEDdetector

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def transform(bytes: bytes,
              model: HEDdetector) -> bytes:
    """transforms an image to a HED map

    Args:
        bytes (bytes): input image
        model (HEDdetector): HED detector model

    Returns:
        bytes: HED map
    """
    # load image
    image = Image.open(io.BytesIO(bytes)).convert("RGB")

    # calculate HED map
    hed_image = model(image)

    # save image to bytes
    output_bytes = io.BytesIO()
    hed_image.save(output_bytes, format='JPEG')

    return output_bytes.getvalue()


class HEDProcessorComponent(TransformComponent):
    """
    Component that calculates the HED map of an image
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

        # load HED detector model
        model = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        model.netNetwork.to(device)

        # calculate HED map
        dataframe['hed_data'] = dataframe['images_data'].apply(lambda x: transform(x, model), meta=('hed_data', 'bytes'))

        return dataframe


if __name__ == "__main__":
    component = HEDProcessorComponent.from_file()
    component.run()
