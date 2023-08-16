"""This component filters images of the dataset based on image size
(minimum image size and maximumn aspect ratio).
"""
import logging

import numpy as np
import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor

logger = logging.getLogger(__name__)


class FilterImageResolutionComponent(PandasTransformComponent):
    """Component that filters images based on height and width."""

    def __init__(self, *_, min_image_dim: int, max_aspect_ratio: float) -> None:
        """
        Args:
            min_image_dim: minimum image dimension.
            max_aspect_ratio: maximum aspect ratio.
        """
        self.min_image_dim = min_image_dim
        self.max_aspect_ratio = max_aspect_ratio

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        width = dataframe["images"]["width"]
        height = dataframe["images"]["height"]
        min_image_dim = np.minimum(width, height)
        max_image_dim = np.maximum(width, height)
        aspect_ratio = max_image_dim / min_image_dim
        mask = (min_image_dim >= self.min_image_dim) & (aspect_ratio <= self.max_aspect_ratio)

        return dataframe[mask]


if __name__ == "__main__":
    executor = PandasTransformExecutor.from_args()
    executor.execute(FilterImageResolutionComponent)
