"""A component that filters text based on the language."""
import logging

import fasttext
import pandas as pd

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class LanguageIdentification:
    """A class for language detection using FastText."""

    def __init__(self, language, model_path: str = "lid.176.ftz"):
        """
        Initializes the LanguageDetect class.

        Args:
           language (str): language to filter on
           model_path (str): The path to the FastText language identification model.
        """
        pretrained_lang_model_weight_path = model_path
        self.language = language
        self.model = fasttext.load_model(pretrained_lang_model_weight_path)

    def predict_lang(self, text: str):
        """
        Detects the language of a text sequence.

        Args:
            text (str): The text for language detection.

        Returns:
            str: The predicted language label.
        """
        predictions = self.model.predict(text, k=1)
        return predictions[0][0]

    def is_language(self, row):
        """Predict if text of a row is written in the defined language."""
        return self.language in self.predict_lang(row["text"])


class LanguageFilterComponent(PandasTransformComponent):
    """Component that filter columns based on provided language."""

    def setup(self, *, language):
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
    component = LanguageFilterComponent.from_args()
    component.run()
