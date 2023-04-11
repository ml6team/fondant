"""
This component filters images of the dataset based on image size (minimum height and width).

Technically, it updates the index of the manifest.
"""
import logging
from typing import Dict, Union

from datasets import Dataset

from express.components.common import FondantManifest, FondantComponent
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def check_min_size(example, min_width, min_height):
    width, height = example["width"], example["height"]

    return width > min_width and height > min_height


class ImageFilterComponent(FondantComponent):
    """
    Component that filters images based on height and width.
    """

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

        # 1) Load one particular data source from the manifest
        logger.info("Loading image dataset...")
        metadata_dataset = manifest.load(
            data_source="images", columns=["index", "width", "height"]
        )

        # 2) Update index by filtering
        logger.info("Filtering dataset...")
        min_width, min_height = args["min_width"], args["min_height"]
        filtered_dataset = metadata_dataset.filter(lambda example: example["width"] > min_width and example["height"] > min_height)
        index_dataset = Dataset.from_dict({"index": filtered_dataset["index"]})

        # 3) Update index of the manifest
        manifest.update_index(index_dataset)

        return manifest


if __name__ == "__main__":
    ImageFilterComponent.run()
