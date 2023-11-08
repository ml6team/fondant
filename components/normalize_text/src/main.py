"""A component that normalizes text."""
import logging
import re
import string
from typing import List

import ftfy
import pandas as pd
from fondant.component import PandasTransformComponent
from utils import is_counter, is_one_word, mainly_uppercase, only_numerical

logger = logging.getLogger(__name__)


def _remove_punctuation(text):
    """Remove punctuation in given text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def _remove_additional_whitespaces(text):
    """
    Text cleaning method from slimpajama approach.
    https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/slimpajama/preprocessing/filter.py
    Apply remove punctuation, and remove consecutive spaces, newlines, tabs in the middle
    and in the beginning / end.
    """
    return re.sub(r"\s+", " ", text.strip())


def normalize_lines(text):
    def any_condition_met(line, discard_condition_functions):
        return any(condition(line) for condition in discard_condition_functions)

    discard_conditions = [mainly_uppercase, only_numerical, is_counter, is_one_word]
    return " ".join(
        [
            line
            for line in text.split("\n")
            if not any_condition_met(line, discard_conditions)
        ],
    )


class NormalizeTextComponent(PandasTransformComponent):
    """Component that normalizes text."""

    def __init__(
        self,
        *args,
        remove_additional_whitespaces: bool,
        apply_nfc: bool,
        normalize_lines: bool,
        do_lowercase: bool,
        remove_punctuation: bool,
    ):
        self.remove_additional_whitespaces = remove_additional_whitespaces
        self.apply_nfc = apply_nfc
        self.normalize_lines = normalize_lines
        self.do_lowercase = do_lowercase
        self.remove_punctuation = remove_punctuation

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
        - Removing of unnecessary whitespaces (e.g. tabs), punctuation
        - Apply line-wise transformations that exclude lines matching specified patterns.
        Patterns include lines that are mainly composed of uppercase characters, lines that consist
        only of numerical characters, lines that are counters (e.g., "3 likes"), and lines
        that contain only one word.

        Args:
            dataframe: Pandas dataframe.

        Returns:
            Pandas dataframe
        """
        if self.normalize_lines:
            dataframe[("text", "data")] = dataframe[("text", "data")].apply(
                normalize_lines,
            )

        if self.do_lowercase:
            dataframe[("text", "data")] = dataframe[("text", "data")].apply(
                lambda x: x.lower(),
            )

        if self.apply_nfc:
            dataframe[("text", "data")] = dataframe[("text", "data")].apply(
                self._do_nfc_normalization,
            )

        if self.remove_punctuation:
            dataframe[("text", "data")] = dataframe[("text", "data")].apply(
                _remove_punctuation,
            )

        if self.remove_additional_whitespaces:
            dataframe[("text", "data")] = dataframe[("text", "data")].apply(
                _remove_additional_whitespaces,
            )

        # remove all empty rows
        dataframe = dataframe[dataframe[("text", "data")].astype(bool)]

        return dataframe
