"""
This component estimates the code to comments ratio and filters instances between two chosen
minimum and maximum values.
"""
import logging

import pandas as pd
from fondant.component import PandasTransformComponent
from utils.text_extraction import get_comments_to_code_ratio

logger = logging.getLogger(__name__)


class FilterCommentsComponent(PandasTransformComponent):
    """Component that filters instances based on code to comments ratio.

    Args:
        min_comments_ratio: The minimum code to comment ratio
        max_comments_ratio: The maximum code to comment ratio
    """

    def __init__(self, *args, min_comments_ratio: float, max_comments_ratio: float) -> None:
        self.min_comments_ratio = min_comments_ratio
        self.max_comments_ratio = max_comments_ratio

    def transform(
            self,
            dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        comment_to_code_ratio = dataframe["code"]["content"].apply(get_comments_to_code_ratio)
        mask = comment_to_code_ratio.between(self.min_comments_ratio, self.max_comments_ratio)
        return dataframe[mask]
