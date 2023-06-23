"""This component that segments images using a model from the Hugging Face hub."""
import io
import logging
import typing as t

import numpy as np
import pandas as pd
import torch
from palette import palette
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, BatchFeature, SegformerImageProcessor

from fondant.component import PandasTransformComponent

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
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8,
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
        """Load the bytestring as an image."""
        bytes_ = io.BytesIO(img)
        return Image.open(bytes_).convert("RGB")

    def transform(img: Image) -> BatchFeature:
        """Transform the image to a tensor using a clip processor and move it to the specified
        device.
        """
        return processor(images=img, return_tensors="pt").to(device)

    return transform(load(image))["pixel_values"]


@torch.no_grad()
def segment_image_batch(image_batch: pd.DataFrame, *, model: AutoModelForSemanticSegmentation,
                        processor: SegformerImageProcessor) -> pd.Series:
    """Embed a batch of images."""
    input_batch = torch.cat(image_batch.tolist())
    output_batch = model(input_batch)
    post_processed_batch = processor.post_process_semantic_segmentation(
        output_batch,
    )
    segmentations_batch = [convert_to_rgb(seg.cpu().numpy()) for seg in post_processed_batch]
    return pd.Series(segmentations_batch, index=image_batch.index)


class SegmentImagesComponent(PandasTransformComponent):
    """Component that segments images using a model from the Hugging Face hub."""

    def setup(self, model_id: str, batch_size: int) -> None:
        """
        Args:
            model_id: id of the model on the Hugging Face hub
            batch_size: batch size to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(self.device)

        self.batch_size = batch_size

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        images = dataframe["images"]["data"].apply(
            process_image,
            processor=self.processor,
            device=self.device,
        )

        results: t.List[pd.Series] = []
        for batch in np.split(images, np.arange(self.batch_size, len(images), self.batch_size)):
            if not batch.empty:
                results.append(
                    segment_image_batch(
                        batch,
                        model=self.model,
                        processor=self.processor,
                    ).T,
                )

        return pd.concat(results).to_frame(name=("segmentations", "data"))


if __name__ == "__main__":
    component = SegmentImagesComponent.from_args()
    component.run()
