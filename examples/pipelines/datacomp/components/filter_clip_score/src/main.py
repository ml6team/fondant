import logging
import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor

logger = logging.getLogger(__name__)

class FilterTextComplexity(PandasTransformComponent):
    """
    Component that filters rows based on clip scores
    """

    def __init__(self, pct_threshold: float, *args, **kwargs):
        self.pct_threshold = pct_threshold

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering on clip scores...")
        logger.info(f"Initial length: {len(dataframe)}")

        clip_scores = dataframe["image_text"]["clip_l14_similarity_score"]
        sorted_clip_scores = clip_scores.sort_values(ascending=False)
        threshold_idx = int(len(sorted_clip_scores) * self.pct_threshold)
        threshold = sorted_clip_scores.iloc[threshold_idx]
        logger.info(f"Clip score Threshold: {threshold}")

        mask = clip_scores > threshold
        filtered_dataframe = dataframe[mask]
        logger.info(f"Final length: {len(filtered_dataframe)} ({len(filtered_dataframe) / len(dataframe):.2f})")

        return filtered_dataframe


