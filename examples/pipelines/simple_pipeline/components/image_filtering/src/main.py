"""
This component filters images of the dataset based on image size (minimum height and width).
"""
import logging
from typing import Dict

import dask.dataframe as dd

from express.dataset import FondantComponent
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ImageFilterComponent(FondantComponent):
    """
    Component that filters images based on height and width.
    """
    def process(self, dataset: dd.DataFrame, args: Dict) -> dd.DataFrame:
        """
        Args:
            dataset
            args: args to pass to the function
        
        Returns:
            dataset
        """
        logger.info("Filtering dataset...")
        min_width, min_height = args.min_width, args.min_height
        filtered_dataset = dataset.filter(lambda example: example["images_width"] > min_width and example["images_height"] > min_height)
        
        return filtered_dataset


if __name__ == "__main__":
    component = ImageFilterComponent()
    component.run()