"""This component filters code based on a set of metadata associated with it."""
import logging

import pandas as pd
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class FilterLineLengthComponent(PandasTransformComponent):
    """
    This component filters code based on a set of metadata associated with it:
    average line length, maximum line length and alphanum fraction.
    """

    def __init__(self, *_,
                 avg_line_length_threshold: int,
                 max_line_length_threshold: int,
                 alphanum_fraction_threshold: float,
                 ) -> None:
        """
        Args:
            avg_line_length_threshold: Threshold for average line length to filter on
            max_line_length_threshold: Threshold for max line length to filter on
            alphanum_fraction_threshold: Alphanum fraction to filter on.
        """
        self.avg_line_length_threshold = avg_line_length_threshold
        self.max_line_length_threshold = max_line_length_threshold
        self.alphanum_fraction_threshold = alphanum_fraction_threshold

    def transform(
            self,
            dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        return dataframe[
            (dataframe["code_avg_line_length"] > self.avg_line_length_threshold)
            & (dataframe["code_max_line_length"] > self.max_line_length_threshold)
            & (dataframe["code_alphanum_fraction"] > self.alphanum_fraction_threshold)
            ]
