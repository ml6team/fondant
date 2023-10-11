"""This component that embeds images using a model from the Hugging Face hub."""
import io
import logging
import os
import typing as t

import numpy as np
import pandas as pd
import torch
from fondant.component import PandasTransformComponent
from PIL import Image
from transformers import BatchEncoding, CLIPProcessor, CLIPVisionModelWithProjection

logger = logging.getLogger(__name__)

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


def process_image_batch(
    images: np.ndarray,
    *,
    processor: CLIPProcessor,
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
        """Transform the image to a tensor using a clip processor and move it to the specified
        device.
        """
        # Edge case: https://github.com/huggingface/transformers/issues/21638
        if img.width == 1 or img.height == 1:
            img = img.resize((224, 224))

        return processor(images=img, return_tensors="pt").to(device)

    return [transform(load(image))["pixel_values"] for image in images]


@torch.no_grad()
def embed_image_batch(
    image_batch: t.List[torch.Tensor],
    *,
    model: CLIPVisionModelWithProjection,
    index: pd.Series,
) -> pd.Series:
    """Embed a batch of images."""
    input_batch = torch.cat(image_batch)
    output_batch = model(input_batch)
    embeddings_batch = output_batch.image_embeds.cpu().tolist()
    return pd.Series(embeddings_batch, index=index)


class EmbedImagesComponent(PandasTransformComponent):
    """Component that embeds images using a CLIP model from the Hugging Face hub."""

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
        logger.info("device used is %s", self.device)

        logger.info("Initialize model '%s'", model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(
            self.device,
        )
        logger.info("Model initialized")

        self.batch_size = batch_size

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        images = dataframe["images"]["data"]

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
                embeddings = embed_image_batch(
                    image_tensors,
                    model=self.model,
                    index=batch.index,
                ).T
                results.append(embeddings)

        return pd.concat(results).to_frame(name=("embeddings", "data"))
