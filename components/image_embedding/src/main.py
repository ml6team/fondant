"""This component that embeds images using a model from the Hugging Face hub."""
import io
import logging
import typing as t

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


def process_image(image: bytes, *, processor: CLIPProcessor, device: str) -> torch.Tensor:
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

    def transform(img: Image) -> torch.Tensor:
        """Transform the image to a tensor using a clip processor and move it to the specified
        device.
        """
        return processor(images=img, return_tensors="pt").to(device)

    return transform(load(image))["pixel_values"]


@torch.no_grad()
def embed_image_batch(image_batch: pd.DataFrame, *, model: CLIPVisionModelWithProjection) -> \
        pd.Series:
    """Embed a batch of images."""
    input_batch = torch.cat(image_batch.tolist())
    output_batch = model(input_batch)
    embeddings_batch = output_batch.image_embeds.cpu().tolist()
    return pd.Series(embeddings_batch, index=image_batch.index)


class EmbedImagesComponent(PandasTransformComponent):
    """Component that captions images using a model from the Hugging Face hub."""

    def setup(
        self,
        *,
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
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device)
        logger.info("Model initialized")

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
                results.append(embed_image_batch(batch, model=self.model).T)

        return pd.concat(results).to_frame(name=("embeddings", "data"))


if __name__ == "__main__":
    component = EmbedImagesComponent.from_args()
    component.run()
