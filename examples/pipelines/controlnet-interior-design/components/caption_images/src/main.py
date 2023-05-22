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

from transformers import AutoProcessor, BlipForConditionalGeneration
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
def flatten(list_of_tuples):
    tuple_of_lists = zip(*list_of_tuples)
    ids, captions = (itertools.chain(*lst) for lst in tuple_of_lists)
    return (
        pd.DataFrame({"id": ids, "captions_text": captions})
        .astype("string")
        .set_index("id")
    )


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
        logger.info(f"Device: {device}")

        processor = AutoProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)

        # load and transform the images
        loaded_images = (
            (index, load(row["images_data"])) for index, row in dataframe.iterrows()
        )
        transformed_images = (
            (index, transform(image, processor, device))
            for index, image in loaded_images
        )

        # batch images together
        batches = (
            zip(*batch) for batch in toolz.partition(batch_size, transformed_images)
        )
        batched_ids, batched_images = zip(*batches)
        batched_images = (collate(batch) for batch in batched_images)

        # caption images
        delayed_model = dask.delayed(model.to(device))
        captions = [
            (id_batch, caption(image_batch, delayed_model, processor, max_new_tokens))
            for id_batch, image_batch in zip(batched_ids, batched_images)
        ]

        # join lists into a single Dask delayed object
        captions = flatten(captions)
        captions_df = dd.from_delayed(
            captions, meta=pd.DataFrame(columns=["captions_text"], dtype="string")
        )

        return captions_df


if __name__ == "__main__":
    component = CaptionImagesComponent.from_file()
    component.run()
