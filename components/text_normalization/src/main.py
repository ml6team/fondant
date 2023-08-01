"""A component that normalizes text."""
import logging
import re
import string
from typing import List

import ftfy
import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor

logger = logging.getLogger(__name__)


def clean(text, remove_punctuation=True):
    """
    Text cleaning method from slimpajama approach.
    https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/slimpajama/preprocessing/filter.py
    Apply remove punctuation, and remove consecutive spaces, newlines, tabs in the middle
    and in the beginning / end.

    Args:
         - text: text to be cleaned
    """
    # remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    return text

def remove_noisy_lines(text, language):
    """
    !!! and note that they require adaptation across languages !!!
    • If it is short (≤ 10 words) and matches a pattern (edit):
        - At the beginning of the line (e.g. sign-in);
        - At the end of the line (e.g. Read more...);
        - Anywhere in the line (e.g. items in cart).
    """
    language  + "bad_patterns.txt"

    def any_condition_met(line, discard_condition_functions):
        return any(condition(line) for condition in discard_condition_functions)

    return " ".join([line for line in text.split("\n") if not any_condition_met])

class TextNormalizationComponent(PandasTransformComponent):
    """Component that normalizes text."""

    def __init__(self, *args, apply_nfc: bool, do_lowercase: bool, characters_to_remove: List[str]):
        self.apply_nfc = apply_nfc
        self.do_lowercase = do_lowercase
        self.characters_to_remove = characters_to_remove
        self.default_cleaning = True

    @staticmethod
    def _do_nfc_normalization(text: str):
        """Apply nfc normalization to the text of the dataframe."""
        return ftfy.fix_text(text, normalization="NFC")

    @staticmethod
    def _remove_patterns(regex_patterns: List[str], text: str):
        """Remove each regex pattern in the provided string."""
        for pattern in regex_patterns:
            text = re.sub(pattern, "", text)
        return text

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization transformations. The component is capable of:
        - NFC normalization
        - Lowercasing
        - Removing of regex patterns.

        Args:
            dataframe: Pandas dataframe.

        Returns:
            Pandas dataframe
        """
        dataframe[("text", "data")] = dataframe["text"]["data"].apply(remove_noisy_lines)

        if self.apply_nfc:
            dataframe[("text", "data")] = dataframe["text"]["data"].apply(lambda x: self._do_nfc_normalization(x))

        if self.do_lowercase:
            dataframe[("text", "data")] = dataframe["text"]["data"].apply(lambda x: x.lower())

        if self.default_cleaning:
            dataframe[("text", "data")] = dataframe["text"]["data"].apply(clean)

        if len(self.characters_to_remove) > 0:
            dataframe[("text", "data")] = dataframe["text"]["data"].apply(
                lambda x: self._remove_patterns(
                    self.characters_to_remove, x,
                ),
            )

        return dataframe


if __name__ == "__main__":
    executor = PandasTransformExecutor.from_args()
    executor.execute(TextNormalizationComponent)
