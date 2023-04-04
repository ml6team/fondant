"""
This component loads a seed dataset from the hub and creates the initial manifest.
"""
import logging
import sys
from typing import Optional, Union, Dict

from datasets import Dataset, load_dataset

import express
from express.components.hf_datasets_components import (
    HFDatasetsLoaderComponent,
    HFDatasetsDatasetDraft,
)
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def create_image_metadata(batch):
    images = batch["image"]

    # add width, height and byte size columns
    widths, heights = zip(*[image.size for image in images])
    batch["width"] = list(widths)
    batch["height"] = list(heights)
    batch["byte_size"] = [sys.getsizeof(image.tobytes()) for image in images]

    return batch


class LoadFromHubComponent(HFDatasetsLoaderComponent):
    """Component that loads a dataset from the hub and creates the initial manifest."""

    data_sources_in = None
    data_sources_out = [("images", express.Image), ("captions", express.Text)]

    @classmethod
    def load(
        cls, extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ) -> HFDatasetsDatasetDraft:
        """
        An example function showcasing the data loader component using Express functionalities
        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function (e.g. seed data source)
        Returns:
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output manifest
        """

        # 1) Create data source
        logger.info("Loading caption dataset from the hub...")
        # TODO perhaps leverage streaming
        dataset = load_dataset(extra_args["dataset_name"], split="train")

        # 2) Create an example index
        logger.info("Creating index...")
        index_list = [f"image_{idx}" for idx in range(len(dataset))]

        # 3) Create dataset draft (manifest without metadata)
        # We store the index itself also as a HF Dataset
        logger.info("Creating draft...")
        index_dataset = Dataset.from_dict({"index": index_list})
        image_dataset = dataset.remove_columns(["text"]).add_column(
            name="index", column=index_list
        )
        text_dataset = dataset.remove_columns(["image"]).add_column(
            name="index", column=index_list
        )
        image_dataset = image_dataset.map(
            create_image_metadata,
            batched=True,
            batch_size=extra_args["batch_size"],
        )
        data_sources = {
            "images": image_dataset,
            "captions": text_dataset,
        }
        dataset_draft = HFDatasetsDatasetDraft(
            index=index_dataset, data_sources=data_sources
        )

        return dataset_draft


if __name__ == "__main__":
    LoadFromHubComponent.run()
