"""A component that filters out text passages which are to short."""
import logging

import pandas as pd

from fondant.component import PandasTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class TextLengthFilterComponent(PandasTransformComponent):
    """A component that filters out text passages which are to short."""

    def setup(self, *, min_length: int):
        self.min_length = min_length

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out text passages which are to short.

        Args:
            dataframe: Pandas dataframe.

        Returns:
            Pandas dataframe.
        """
        mask = (
                dataframe["text"]["data"].apply(lambda x: len(x.split()))
                >= self.min_length
        )
        dataframe = dataframe[mask]
        return dataframe


if __name__ == "__main__":
    component = TextLengthFilterComponent.from_args()
    component.run()
