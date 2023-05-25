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


@dask.delayed
def load(example):
    bytes = io.BytesIO(example)
    image = Image.open(bytes).convert("RGB")
    return image


@dask.delayed
# @torch.no_grad()
def transform(image, model):
    outputs = model(image, hand_and_face=True)
    return outputs

@dask.delayed
# @torch.no_grad()
def transform_canny(image):
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(np.array(image), low_threshold, high_threshold) 
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


class OpenposeComponent(TransformComponent):
    """
    Component that segments images using a model from the Hugging Face hub.
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            batch_size: batch size to use

        Returns:
            Dask dataframe
        """
        # model = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

        # caption images
        # delayed_model = dask.delayed(model.to(device))
        dataframe['openpose_data'] = images['images_data'].apply(lambda x: transform_canny(load(x)))

        return dataframe


if __name__ == "__main__":
    component = OpenposeComponent.from_file()
    component.run()
