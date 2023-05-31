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
from controlnet_aux import MLSDdetector

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def transform(bytes: bytes,
              model: MLSDdetector) -> bytes:
    """transforms an image to a MLSD map

    Args:
        bytes (bytes): input image
        model (MLSDdetector): MLSD detector model

    Returns:
        bytes: MLSD map
    """
    # load image
    image = Image.open(io.BytesIO(bytes)).convert("RGB")

    # calculate MLSD map
    mlsd_image = model(image)

    # save image to bytes
    output_bytes = io.BytesIO()
    mlsd_image.save(output_bytes, format='JPEG')

    return output_bytes.getvalue()


class MLSDProcessorComponent(TransformComponent):
    """
    Component that calculates the MLSD map of an image
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

        # load MLSD detector model
        model = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        model.to(device)

        # calculate mlsd map
        dataframe['mlsd_data'] = dataframe['images_data'].apply(lambda x: transform(x, model), meta=('mlsd_data', 'bytes'))

        return dataframe


if __name__ == "__main__":
    component = MLSDProcessorComponent.from_file()
    component.run()
