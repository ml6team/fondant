"""
This component that embeds images using a model from the Hugging Face hub.
"""
import io
import itertools
import logging
import toolz
import typing as t

from PIL import Image
import torch
import dask
import pandas as pd
import dask.dataframe as dd
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@dask.delayed
def load(example: bytes) -> Image:
    """
    Load the bytestring as an image

    Args:
        example: A byte string representing the image.

    Returns:
        The loaded image.
    """
    bytes = io.BytesIO(example)
    image = Image.open(bytes).convert("RGB")
    return image


@dask.delayed
def transform(image: Image, processor: CLIPProcessor, device: str) -> torch.Tensor:
    """
    Transform the image to a tensor using a clip processor and move it to the specified device.

    Args:
        image: The input image.
        processor: The processor object for transforming the image.
        device: The device to move the transformed image to.

    Returns:
        The transformed image.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    return inputs


@dask.delayed
def collate(examples: t.List[t.Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Collate a list of examples into a single torch.Tensor.

    Args:
        examples: A list of examples, where each example is a dictionary containing a
         "pixel_values" tensor.

    Returns:
        torch.Tensor: The concatenated tensor of pixel values from all examples.
    """
    return torch.cat([ex["pixel_values"] for ex in examples])


@dask.delayed
@torch.no_grad()
def embed(
    batch: torch.Tensor, model: CLIPVisionModelWithProjection
) -> t.List[t.List[float]]:
    """
    Embed a batch of images using a given model.

    Args:
        batch: The batch of images.
        model: The embedding model.

    Returns:
        The embeddings as a list of lists.
    """
    # embed to get (batch_size, hidden_size) embeddings
    outputs = model(batch)
    image_embeds = outputs.image_embeds

    # flatten into list of embeddings
    embeddings = image_embeds.cpu().tolist()

    return embeddings


@dask.delayed
def flatten(lst: t.List[t.List[t.Any]]) -> pd.Series:
    """
    Flatten a nested list into a pandas Series.

    Parameters:
        lst: The nested list to flatten.

    Returns:
        The flattened Series.
    """
    return pd.Series(itertools.chain(*lst))


class EmbedImagesComponent(TransformComponent):
    """
    Component that captions images using a model from the Hugging Face hub.
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
        logger.info("device used is %s", device)

        logger.info("Initialize model '%s'", model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPVisionModelWithProjection.from_pretrained(model_id)
        model.to(device)

        logger.info("Model initialized")

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

        # embed images
        gpu_model = dask.delayed(model.to(device))
        embeddings = [embed(batch, gpu_model) for batch in batches]

        # join lists into a single Dask delayed object
        embeddings = flatten(embeddings)
        delayed_series = dd.from_delayed(embeddings, meta=pd.Series(dtype="object"))
        embeddings_df = delayed_series.to_frame(name="embeddings_data")

        return embeddings_df


if __name__ == "__main__":
    component = EmbedImagesComponent.from_file()
    component.run()
