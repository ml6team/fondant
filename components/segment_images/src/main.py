"""
This component that segments images using a model from the Hugging Face hub.
"""
import io
import logging
import typing as t

import dask.dataframe as dd
from PIL import Image
import pandas as pd
import numpy as np
from transformers import BatchFeature, SegformerImageProcessor, AutoModelForSemanticSegmentation
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


def process_image(image: bytes, *, processor: SegformerImageProcessor, device: str) -> torch.Tensor:
    """
    Process the image to a tensor.

    Args:
        image: The input image as a byte string.
        processor: The processor object for transforming the image.
        device: The device to move the transformed image to.
    """
    def load(img: bytes) -> Image:
        """Load the bytestring as an image"""
        bytes_ = io.BytesIO(img)
        return Image.open(bytes_).convert("RGB")

    def transform(img: Image) -> BatchFeature:
        """
        Transform the image to a tensor using a clip processor and move it to the specified device.
        """
        return processor(images=img, return_tensors="pt").to(device)

    return transform(load(image))["pixel_values"]


@torch.no_grad()
def segment_image_batch(image_batch: pd.DataFrame, *, model: AutoModelForSemanticSegmentation,
                        processor: SegformerImageProcessor) -> pd.Series:
    """Embed a batch of images"""
    input_batch = torch.cat(image_batch.tolist())
    output_batch = model(input_batch)
    post_processed_batch = processor.post_process_semantic_segmentation(
        output_batch
    )
    segmentations_batch = [convert_to_rgb(seg.cpu().numpy()) for seg in post_processed_batch]
    return pd.Series(segmentations_batch, index=image_batch.index)


def segment_images(
        images: pd.Series,
        *,
        model: AutoModelForSemanticSegmentation,
        processor: SegformerImageProcessor,
        batch_size: int,
        device: str,
):
    """Segment a pandas series of images"""
    images = images.apply(process_image, processor=processor, device=device)
    results: t.List[pd.Series] = []
    for batch in np.split(images, np.arange(batch_size, len(images), batch_size)):
        if not batch.empty:
            results.append(
                segment_image_batch(
                    batch,
                    model=model,
                    processor=processor,
                ).T
            )
    return pd.concat(results).to_frame()


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
        logger.info(f"Device: {device}")

        processor = SegformerImageProcessor.from_pretrained(model_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id)

        dataframe = dataframe["images_data"].map_partitions(
            segment_images,
            model=model,
            processor=processor,
            batch_size=batch_size,
            device=device,
            meta={0: object}
        )
        dataframe.columns = ["segmentations_data"]

        return dataframe


if __name__ == "__main__":
    component = SegmentImagesComponent.from_file()
    component.run()
