"""A component that filters text based on the language."""
import logging

import fasttext
import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor

logger = logging.getLogger(__name__)

MODEL_PATH = "lid.176.ftz"


class LanguageIdentification:
    """A class for language detection using FastText."""

    def __init__(self, language: str):
        """
        Initializes the LanguageDetect class.

        Args:
           language (str): language to filter on
        """
        self.language = language
        self.model = fasttext.load_model(MODEL_PATH)

    def predict_lang(self, text: str):
        """
        Detects the language of a text sequence.

        Args:
            text (str): The text for language detection.

        Returns:
            str: The predicted language label.
        """
        if "\n" in text:
            text = text.replace("\n", "")

        # Idea: short text pieces enough to determine the language. Open question: how long is sufficient
        # TODO: ad hoc testing on a large dataset
        if len(text) >= 200:
            text = text[:200]

        predictions = self.model.predict(text, k=1)
        return predictions[0][0]

    def is_language(self, row):
        """Predict if text of a row is written in the defined language."""
        if isinstance(row["webpage"]["html"], str):
            return self.language in self.predict_lang(row["webpage"]["html"])
        else:
            return False


class LanguageFilterComponent(PandasTransformComponent):
    """Component that filter columns based on provided language."""

    def __init__(self, *_, language):
        """Setup language filter component.

        Args:
            language: Only keep text passages which are in the provided language.
        """
        self.lang_detector = LanguageIdentification(language)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            dataframe: Pandas dataframe.

        Returns:
            Pandas dataframe
        """
        mask = dataframe.apply(self.lang_detector.is_language, axis=1)

        return dataframe[mask]


if __name__ == "__main__":
    executor = PandasTransformExecutor.from_args()
    executor.execute(LanguageFilterComponent)
