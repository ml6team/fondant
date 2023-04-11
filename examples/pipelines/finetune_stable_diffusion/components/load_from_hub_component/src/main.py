"""
This component loads a seed dataset from the hub and creates the initial manifest.
"""
import logging
import sys
from typing import Union, Dict

from datasets import Dataset, load_dataset

from express.components.common import FondantManifest, FondantComponent
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


class LoadFromHubComponent(FondantComponent):
    """Component that loads a dataset from the hub and adds it to the manifest."""

    @classmethod
    def process(
        cls,
        manifest: FondantManifest,
        args: Dict[str, Union[str, int, float, bool]],
    ) -> FondantManifest:
        """
        Args:
            manifest: Fondant manifest
            args: args to pass to the function
        
        Returns:
            FondantManifest: output manifest
        """
        # 1) Create data source
        logger.info("Loading caption dataset from the hub...")
        # TODO perhaps leverage streaming
        dataset = load_dataset(args["dataset_name"], split="train")

        # 2) Create index
        logger.info("Creating index...")
        index_list = [f"image_{idx}" for idx in range(len(dataset))]

        # 3) Create data sources
        # We store the index itself also as a HF Dataset
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
            batch_size=args["batch_size"],
        )
        data_sources = {
            "images": image_dataset,
            "captions": text_dataset,
        }
        manifest._create_index(index_dataset)
        manifest.add_data_sources(data_sources)

        return manifest


if __name__ == "__main__":
    LoadFromHubComponent.run()
