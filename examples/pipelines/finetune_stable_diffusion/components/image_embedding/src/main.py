"""
This component that embeds images using a model from the Hugging Face hub.
"""
import io
import itertools
import logging
import toolz

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
def load(example):
    bytes = io.BytesIO(example)
    image = Image.open(bytes).convert("RGB")
    return image


@dask.delayed
def transform(image, processor, device):
    inputs = processor(images=image, return_tensors="pt").to(device)

    return inputs


@dask.delayed
def collate(examples):
    return torch.cat([ex["pixel_values"] for ex in examples])


@dask.delayed
@torch.no_grad()
def embed(batch, model):
    # embed to get (batch_size, hidden_size) embeddings
    outputs = model(batch)
    image_embeds = outputs.image_embeds

    # flatten into list of embeddings
    embeddings = image_embeds.cpu().tolist()

    return embeddings


@dask.delayed
def flatten(lst):
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

        # add index columns
        embeddings_df["id"] = dataframe["id"].reset_index(drop=True)
        embeddings_df["source"] = dataframe["source"].reset_index(drop=True)

        embeddings_df = embeddings_df.reset_index(drop=True)

        return embeddings_df


if __name__ == "__main__":
    component = EmbedImagesComponent.from_file()
    component.run()
