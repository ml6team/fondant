"""
This component that captions images using a model from the Hugging Face hub.
"""
import io
import itertools
import logging
import toolz

from PIL import Image

import dask
import dask.dataframe as dd
import pandas as pd

from transformers import AutoProcessor, AutoModelForCausalLM
import torch

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
def caption(batch, model, processor, max_new_tokens):
    logger.info("Generating caption...")
    generated_ids = model.generate(pixel_values=batch, max_new_tokens=max_new_tokens)
    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_captions


@dask.delayed
def flatten(lst):
    return pd.Series(itertools.chain(*lst))


class CaptionImagesComponent(TransformComponent):
    """
    Component that captions images using a model from the Hugging Face hub.
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
        model_id: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            model_id: id of the model on the Hugging Face hub
            batch_size: batch size to use
            max_new_tokens: maximum token length of each caption

        Returns:
            Dask dataframe
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        print("Length of the dataframe:", len(dataframe))

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

        # caption images
        delayed_model = dask.delayed(model.to(device))
        captions = [
            caption(batch, delayed_model, processor, max_new_tokens)
            for batch in batches
        ]

        # join lists into a single Dask delayed object
        captions = flatten(captions)
        delayed_series = dd.from_delayed(captions, meta=pd.Series(dtype="str"))
        captions_df = delayed_series.to_frame(name="captions_text")

        # add index columns
        captions_df["id"] = dataframe["id"].reset_index(drop=True)
        captions_df["source"] = dataframe["source"].reset_index(drop=True)

        return captions_df


if __name__ == "__main__":
    component = CaptionImagesComponent.from_file()
    component.run()
