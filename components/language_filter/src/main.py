"""A component that filter a provided dataframe based on the language"""
import logging
import dask.dataframe as dd
from fondant.component import DaskTransformComponent
from fondant.logger import configure_logging
import fasttext

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
        return language in self.predict_lang(row["text"])


class LanguageFilterComponent(DaskTransformComponent):
    """Component that filter columns based on provided language"""

    def transform(
            self,
            *,
            dataframe: dd.DataFrame,
            language: str,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe.
            language: Only keep text passages which are in the provided language

        Returns:
            Dask dataframe
        """

        lang_detector = LanguageIdentification()
        mask = dataframe.map_partitions(
            lambda df: df.apply(lambda row: lang_detector.is_language(row, language), axis=1),
            meta=bool)

        return dataframe[mask]


if __name__ == "__main__":
    component = LanguageFilterComponent.from_args()
    component.run()
