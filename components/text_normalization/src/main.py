"""A component that normalizes text."""
import logging
import re
import unicodedata
from typing import List

import pandas as pd

from fondant.component import PandasTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class TextNormalizationComponent(PandasTransformComponent):
    """Component that normalizes text."""

    def setup(self, *, apply_nfc: bool, do_lowercase: bool, characters_to_remove: List[str]):
        self.apply_nfc = apply_nfc
        self.do_lowercase = do_lowercase
        self.characters_to_remove = characters_to_remove

    @staticmethod
    def _do_nfc_normalization(text: str):
        """Apply nfc normalization to the text of the dataframe."""
        return unicodedata.normalize("NFC", text)

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
        if self.apply_nfc:
            dataframe["text"]["data"].apply(lambda x: self._do_nfc_normalization(x))

        if self.do_lowercase:
            dataframe["text"]["data"].apply(lambda x: x.lower())

        if len(self.characters_to_remove) > 0:
            dataframe["text"]["data"].apply(
                lambda x: self._remove_patterns(
                    self.characters_to_remove, x
                )
            )

        return dataframe


if __name__ == "__main__":
    component = TextNormalizationComponent.from_args()
    component.run()
