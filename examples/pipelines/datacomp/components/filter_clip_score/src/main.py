import logging
import dask
import pandas as pd
import typing as t
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)

dask.config.set(scheduler="processes")


class FilterTextComplexity(PandasTransformComponent):
    """
    Component that filters rows based on clip scores
    """

    def __init__(
        self,
        *args,
        pct_threshold: t.Optional[float],
        threshold_score: t.Optional[float],
    ):
        self.pct_threshold = pct_threshold
        self.threshold_score = threshold_score

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering on clip scores...")
        logger.info(f"Initial length: {len(dataframe)}")

        clip_scores = dataframe["imagetext"]["clipl14score"]

        if self.pct_threshold and self.threshold_score is None:
            sorted_clip_scores = clip_scores.sort_values(ascending=False)
            threshold_idx = int(len(sorted_clip_scores) * self.pct_threshold)
            threshold = sorted_clip_scores.iloc[threshold_idx]
        elif self.pct_threshold is None and self.threshold_score:
            threshold = self.threshold_score
        elif self.pct_threshold and self.threshold_score:
            raise ValueError("Only one of pct_threshold or threshold_score can be set")
        else:
            raise ValueError("One of pct_threshold or threshold_score must be set")

        logger.info(f"Clip score Threshold: {threshold}")

        mask = clip_scores > threshold
        filtered_dataframe = dataframe[mask]

        return filtered_dataframe
