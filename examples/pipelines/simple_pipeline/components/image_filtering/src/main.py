"""
This component filters images of the dataset based on image size (minimum height and width).
"""
import logging
from typing import Dict

import dask.dataframe as dd

from fondant.component import FondantTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ImageFilterComponent(FondantTransformComponent):
    """
    Component that filters images based on height and width.
    """

    def transform(self, dataframe: dd.DataFrame, *, min_width: int, min_height: int) -> dd.DataFrame:
        """
        Args:
            df: Dask dataframe
            min_width: minimum width of the resulting images
            min_height: minimum height of the resulting images
        
        Returns:
            Dask dataframe
        """
        logger.info("Length of the dataframe before filtering:", len(dataframe))

        logger.info("Filtering dataset...")
        filtered_df = dataframe[(dataframe["images_width"] > min_width) & (dataframe["images_height"] > min_height)]

        logger.info("Length of the dataframe after filtering:", len(filtered_df))

        return filtered_df


if __name__ == "__main__":
    component = ImageFilterComponent()
    component.run()
