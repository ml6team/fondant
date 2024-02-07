"""This component that captions images using a model from the Hugging Face hub."""
import io
import logging
import os
import typing as t

import numpy as np
import pandas as pd
import torch
from fondant.component import PandasTransformComponent
from PIL import Image
from transformers import BatchEncoding, BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


def process_image_batch(
    images: np.ndarray,
    *,
    processor: BlipProcessor,
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

    def transform(img: Image) -> BatchEncoding:
        """Transform the image to a tensor using a processor and move it to the specified device."""
        # Edge case: https://github.com/huggingface/transformers/issues/21638
        if img.width == 1 or img.height == 1:
            img = img.resize((224, 224))

        return processor(images=img, return_tensors="pt").to(device)

    return [transform(load(image))["pixel_values"] for image in images]


@torch.no_grad()
def caption_image_batch(
    image_batch: t.List[torch.Tensor],
    *,
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    max_new_tokens: int,
    index: pd.Series,
) -> pd.Series:
    """Caption a batch of images."""
    input_batch = torch.cat(image_batch)
    output_batch = model.generate(
        pixel_values=input_batch,
        max_new_tokens=max_new_tokens,
    )
    captions_batch = processor.batch_decode(output_batch, skip_special_tokens=True)

    return pd.Series(captions_batch, index=index)


class CaptionImagesComponent(PandasTransformComponent):
    """Component that captions images using a model from the Hugging Face hub."""

    def __init__(
        self,
        *,
        model_id: str,
        batch_size: int,
        max_new_tokens: int,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        logger.info("Initialize model '%s'", model_id)
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(
            self.device,
        )

        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        images = dataframe["image"]

        results: t.List[pd.Series] = []
        for batch in np.split(
            images,
            np.arange(self.batch_size, len(images), self.batch_size),
        ):
            if not batch.empty:
                image_tensors = process_image_batch(
                    batch,
                    processor=self.processor,
                    device=self.device,
                )
                captions = caption_image_batch(
                    image_tensors,
                    model=self.model,
                    processor=self.processor,
                    max_new_tokens=self.max_new_tokens,
                    index=batch.index,
                ).T
                results.append(captions)

        return pd.concat(results).to_frame(name="caption")
