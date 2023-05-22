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
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch

from palette import palette

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def convert_to_rgb(seg: np.array):
    """
    Converts a 2D segmentation to a RGB one which makes it possible to visualize it.

    Args:
        seg: 2D segmentation map as a NumPy array.

    Returns:
        color_seg: 3D segmentation map contain RGB values for each pixel.
    """
    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8).tobytes()

    return color_seg


@dask.delayed
def load(example):
    bytes = io.BytesIO(example)
    image = Image.open(bytes).convert("RGB")
    return image


@dask.delayed
def transform(image, processor, device):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    return pixel_values, image.size


@dask.delayed
def collate(examples):
    encoding = {}
    encoding["pixel_values"] = torch.cat([ex[0] for ex in examples])
    encoding["image_sizes"] = [ex[1] for ex in examples]
    return encoding


@dask.delayed
@torch.no_grad()
def segment(batch, model, processor):
    outputs = model(batch["pixel_values"])
    segmentations = processor.post_process_semantic_segmentation(
        outputs, target_sizes=batch["image_sizes"]
    )
    # turn into RGB images
    segmentations = [convert_to_rgb(seg.numpy()) for seg in segmentations]

    return segmentations


@dask.delayed
def flatten(lst):
    return pd.Series(itertools.chain(*lst))


class SegmentImagesComponent(TransformComponent):
    """
    Component that segments images using a model from the Hugging Face hub.
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
        model_id: str,
        batch_size: int,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            model_id: id of the model on the Hugging Face hub
            batch_size: batch size to use

        Returns:
            Dask dataframe
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Device:", device)

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id)

        # load and transform the images
        images = dataframe["images_data"]
        loaded_images = [load(image) for image in images]
        transformed_images = [
            transform(image, processor, device) for image in loaded_images
        ]

        # batch images together
        batches = [
            collate(batch)
            for batch in toolz.partition_all(batch_size, transformed_images)
        ]

        # caption images
        delayed_model = dask.delayed(model.to(device))
        segmentations = [segment(batch, delayed_model, processor) for batch in batches]

        # join lists into a single Dask delayed object
        segmentations = flatten(segmentations)
        delayed_series = dd.from_delayed(segmentations, meta=pd.Series(dtype="object"))
        segmentations_df = delayed_series.to_frame(name="segmentations_data")

        # add index columns
        segmentations_df["id"] = dataframe["id"].reset_index(drop=True)
        segmentations_df["source"] = dataframe["source"].reset_index(drop=True)

        segmentations_df = segmentations_df.reset_index(drop=True)

        return segmentations_df


if __name__ == "__main__":
    component = SegmentImagesComponent.from_file()
    component.run()
