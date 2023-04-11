"""
This component filters images of the dataset based on image size (minimum height and width).

Technically, it updates the index of the manifest.
"""
import logging
from typing import Dict

from datasets import Dataset

from express.components.common import Manifest, ExpressTransformComponent
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def check_min_size(example, min_width, min_height):
    width, height = example["width"], example["height"]

    return width > min_width and height > min_height


class ImageFilterComponent(ExpressTransformComponent):
    """
    Class that inherits from Hugging Face data transform.
    """

    @classmethod
    def transform(
        cls,
        manifest: Manifest,
        args = None,
    ) -> Manifest:
        """
        An example function showcasing the data transform component using Express functionalities

        Args:
            manifest
            data (HFDatasetsDataset[TIndex, TData]): express dataset providing access to data of a
             given type
            args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function
        Returns:
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output manifest
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
