"""
This component that embeds images using a model from the Hugging Face hub.
"""
import io
import logging
import typing as t

from PIL import Image
import torch
import numpy as np
import pandas as pd
import dask.dataframe as dd
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
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
        """Load the bytestring as an image"""
        bytes_ = io.BytesIO(img)
        return Image.open(bytes_).convert("RGB")

    def transform(img: Image) -> torch.Tensor:
        """
        Transform the image to a tensor using a clip processor and move it to the specified device.
        """
        return processor(images=img, return_tensors="pt").to(device)

    return transform(load(image))["pixel_values"]


@torch.no_grad()
def embed_image_batch(image_batch: pd.DataFrame, *, model: CLIPVisionModelWithProjection) -> \
        pd.Series:
    """Embed a batch of images"""
    input_batch = torch.cat(image_batch.tolist())
    output_batch = model(input_batch)
    embeddings_batch = output_batch.image_embeds.cpu().tolist()
    return pd.Series(embeddings_batch, index=image_batch.index)


def embed_images(
    images: pd.Series,
    *,
    model: CLIPVisionModelWithProjection,
    processor: CLIPProcessor,
    batch_size: int,
    device: str
) -> pd.DataFrame:
    """Embed a pandas series of images."""
    images = images.apply(process_image, processor=processor, device=device)
    results: t.List[pd.Series] = []
    for batch in np.split(images, np.arange(batch_size, len(images), batch_size)):
        if not batch.empty:
            results.append(embed_image_batch(batch, model=model).T)
    return pd.concat(results).to_frame()


class EmbedImagesComponent(TransformComponent):
    """
    Component that captions images using a model from the Hugging Face hub.
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
        *,
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
        logger.info("device used is %s", device)

        logger.info("Initialize model '%s'", model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPVisionModelWithProjection.from_pretrained(model_id)
        model.to(device)
        logger.info("Model initialized")

        dataframe = dataframe["images_data"].map_partitions(
            embed_images,
            model=model,
            processor=processor,
            batch_size=batch_size,
            device=device,
            meta={0: object}
        )
        dataframe.columns = ["embeddings_data"]

        return dataframe


if __name__ == "__main__":
    component = EmbedImagesComponent.from_file()
    component.run()
