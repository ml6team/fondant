"""This component filters code based on a set of metadata associated with it."""
import logging

import dask.dataframe as dd

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class FilterLineLengthComponent(TransformComponent):
    """
    This component filters code based on a set of metadata associated with it:
    average line length, maximum line length and alphanum fraction.
    """

    def transform(
        self,
        *,
        dataframe: dd.DataFrame,
        avg_line_length_threshold: int,
        max_line_length_threshold: int,
        alphanum_fraction_threshold: float
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            avg_line_length_threshold: Threshold for average line length to filter on
            max_line_length_threshold: Threshold for max line length to filter on
            alphanum_fraction_threshold: Alphanum fraction to filter on
        Returns:
            Filtered dask dataframe.
        """
        filtered_df = dataframe[
            (dataframe["code_avg_line_length"] > avg_line_length_threshold)
            & (dataframe["code_max_line_length"] > max_line_length_threshold)
            & (dataframe["code_alphanum_fraction"] > alphanum_fraction_threshold)
        ]

        return filtered_df


if __name__ == "__main__":
    component = FilterLineLengthComponent.from_args()
    component.run()