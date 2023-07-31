"""A component that filters out text based on their length."""
import logging

import fasttext
import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor

logger = logging.getLogger(__name__)


class TextLengthFilterComponent(PandasTransformComponent):
    """A component that filters out text based on their length."""

    def __init__(self, *_, min_characters_length: int, min_words_length: int):
        """Setup component.

        Args:
            min_characters_length: minimum number of characters
            min_words_length: minimum number of words.
        """
        self.min_characters_length = min_characters_length
        self.min_words_length = min_words_length

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out text based on their length.

        Args:
            dataframe: Pandas dataframe.

        Returns:
            Pandas dataframe.
        """
        dataframe[("webpage", "html")] = dataframe[("webpage", "html")].apply(
            lambda x: str(x) if x else "None"
        )

        caption_num_words = dataframe["webpage"]["html"].apply(
            lambda x: len(fasttext.tokenize(x))
        )
        caption_num_chars = dataframe["webpage"]["html"].apply(len)

        mask = (caption_num_words >= self.min_words_length) & (
            caption_num_chars >= self.min_characters_length
        )
        dataframe = dataframe[mask]
        return dataframe


if __name__ == "__main__":
    executor = PandasTransformExecutor.from_args()
    executor.execute(TextLengthFilterComponent)
