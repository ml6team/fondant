"""A component that filter a provided dataframe based on the language."""
import logging

import fasttext
import pandas as pd

from fondant.component import PandasTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class LanguageIdentification:
    """A class for language detection using FastText."""

    def __init__(self, model_path: str = "lid.176.ftz"):
        """
        Initializes the LanguageDetect class.

        Args:
           model_path (str): The path to the FastText language identification model.
        """
        pretrained_lang_model_weight_path = model_path
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

    def is_language(self, row, language):
        """Predict if text of a row is written in the defined language."""
        return language in self.predict_lang(row["text"])


class LanguageFilterComponent(PandasTransformComponent):
    """Component that filter columns based on provided language."""

    def setup(self, *args, **kwargs):
        """Setup language filter component."""
        self.lang_detector = LanguageIdentification()

    def transform(
            self,
            dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe.
            language: Only keep text passages which are in the provided language.

        Returns:
            Dask dataframe
        """
        language = self.user_arguments["language"]
        mask = dataframe.apply(
            lambda row: self.lang_detector.is_language(row, language), axis=1)

        return dataframe[mask]


if __name__ == "__main__":
    component = LanguageFilterComponent.from_args()
    component.run()
