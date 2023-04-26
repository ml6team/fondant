"""
This component filters images of the dataset based on image size (minimum height and width).
"""
import logging
from typing import Dict

import dask.dataframe as dd

from fondant.dataset import FondantComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ImageFilterComponent(FondantComponent):
    """
    Component that filters images based on height and width.
    """

    def transform(self, df: dd.DataFrame, args: Dict) -> dd.DataFrame:
        """
        Args:
            df: Dask dataframe
            args: args to pass to the function
        
        Returns:
            dataset
        """
        logger.info("Filtering dataset...")
        min_width, min_height = args.min_width, args.min_height
        filtered_df = df[(df["images_width"] > min_width) & (df["images_height"] > min_height)]

        return filtered_df


if __name__ == "__main__":
    component = ImageFilterComponent()
    component.run()
