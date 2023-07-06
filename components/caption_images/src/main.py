"""This component that captions images using a model from the Hugging Face hub."""
import io
import logging
import typing as t

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import BatchEncoding, BlipForConditionalGeneration, BlipProcessor

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


def process_image(image: bytes, *, processor: BlipProcessor, device: str) -> torch.Tensor:
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

    def transform(img: Image) -> BatchEncoding:
        """Transform the image to a tensor using a processor and move it to the specified device."""
        return processor(images=img, return_tensors="pt").to(device)

    return transform(load(image))["pixel_values"]


def caption_image_batch(
    image_batch: pd.DataFrame,
    *,
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    max_new_tokens: int,
) -> pd.Series:
    """Caption a batch of images."""
    input_batch = torch.cat(image_batch.tolist())
    output_batch = model.generate(pixel_values=input_batch, max_new_tokens=max_new_tokens)
    captions_batch = processor.batch_decode(output_batch, skip_special_tokens=True)

    return pd.Series(captions_batch, index=image_batch.index)


class CaptionImagesComponent(PandasTransformComponent):
    """Component that captions images using a model from the Hugging Face hub."""

    def setup(self, *, model_id: str, batch_size: int, max_new_tokens: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

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
                    caption_image_batch(
                        batch,
                        model=self.model,
                        processor=self.processor,
                        max_new_tokens=self.max_new_tokens,
                    ).T,
                )

        return pd.concat(results).to_frame(name=("captions", "text"))


if __name__ == "__main__":
    component = CaptionImagesComponent.from_args()
    component.run()
