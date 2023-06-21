"""This component filters images of the dataset based on image size (minimum height and width)."""
import logging

import numpy as np
import pandas as pd

from fondant.component import DaskTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ImageFilterComponent(DaskTransformComponent):
    """Component that filters images based on height and width."""

    def transform(
        self, *, dataframe: pd.DataFrame, min_image_dimension: int, max_aspect_ratio: float,
    ) -> pd.DataFrame:
        """
        Args:
            dataframe: Pandas dataframe
            min_width: min width to filter on
            min_height: min height to filter on.

        Returns:
            Pandas dataframe
        """
        logger.info("Filtering dataframe...")

        min_image_dim = np.minimum(dataframe.original_width, dataframe.original_height)
        max_image_dim = np.maximum(dataframe.original_width, dataframe.original_height)
        aspect_ratio = max_image_dim / min_image_dim
        image_mask = (min_image_dim >= min_image_dimension) & (aspect_ratio <= max_aspect_ratio)

        filtered_df = dataframe[image_mask] 

        return filtered_df


if __name__ == "__main__":
    component = ImageFilterComponent.from_args()
    component.run()
