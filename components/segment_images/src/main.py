"""This component that segments images using a model from the Hugging Face hub."""
import io
import logging
import os
import typing as t

import numpy as np
import pandas as pd
import torch
from fondant.component import PandasTransformComponent
from palette import palette
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, BatchFeature, SegformerImageProcessor

logger = logging.getLogger(__name__)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_CUDNN_V8_API_DISABLED'] = "1"


def convert_to_rgb(seg: np.array) -> bytes:
    """
    Converts a 2D segmentation to an RGB one which makes it possible to visualize it.

    Args:
        seg: 2D segmentation map as a NumPy array.

    Returns:
        color_seg: the RGB segmentation map as a binary string
    """
    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8,
    )  # height, width, 3

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    image = Image.fromarray(color_seg).convert('RGB')

    crop_bytes = io.BytesIO()
    image.save(crop_bytes, format="JPEG")

    return crop_bytes.getvalue()


def process_image_batch(
        images: np.ndarray,
        *,
        processor: SegformerImageProcessor,
        device: str,
) -> t.List[torch.Tensor]:
    """
    Process image in batches to a list of tensors.

    Args:
        images: The input images as a numpy array containing byte strings.
        processor: The processor object for transforming the image.
        device: The device to move the transformed image to.
    """

    def load(img: bytes) -> Image:
        """Load the bytestring as an image."""
        bytes_ = io.BytesIO(img)
        return Image.open(bytes_).convert("RGB")

    def transform(img: Image) -> BatchFeature:
        """Transform the image to a tensor using a clip processor and move it to the specified
        device.
        """
        # Edge case: https://github.com/huggingface/transformers/issues/21638
        if img.width == 1 or img.height == 1:
            img = img.resize((224, 224))

        return processor(images=img, return_tensors="pt").to(device)

    return [transform(load(image))["pixel_values"] for image in images]


@torch.no_grad()
def segment_image_batch(
        image_batch: t.List[torch.Tensor],
        *,
        model: AutoModelForSemanticSegmentation,
        processor: SegformerImageProcessor,
        index: pd.Series,
) -> pd.Series:
    """Embed a batch of images."""
    input_batch = torch.cat(image_batch)
    output_batch = model(input_batch)
    post_processed_batch = processor.post_process_semantic_segmentation(
        output_batch,
    )
    segmentations_batch = [convert_to_rgb(seg.cpu().numpy()) for seg in post_processed_batch]
    return pd.Series(segmentations_batch, index=index)


class SegmentImagesComponent(PandasTransformComponent):
    """Component that segments images using a model from the Hugging Face hub."""

    def __init__(
            self,
            *_,
            model_id: str,
            batch_size: int,
    ):
        """
        Args:
            model_id: id of the model on the Hugging Face hub
            batch_size: batch size to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        logger.info("Initialize model '%s'", model_id)
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(self.device)

        self.batch_size = batch_size

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        images = dataframe["images"]["data"]

        results: t.List[pd.Series] = []
        for batch in np.split(images, np.arange(self.batch_size, len(images), self.batch_size)):
            if not batch.empty:
                image_tensors = process_image_batch(
                    batch,
                    processor=self.processor,
                    device=self.device,
                )

                segmentations = segment_image_batch(
                    image_tensors,
                    model=self.model,
                    processor=self.processor,
                    index=batch.index,
                ).T

                results.append(segmentations)

        return pd.concat(results).to_frame(name=("segmentations", "data"))
