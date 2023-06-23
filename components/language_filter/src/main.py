"""A component that detects and redacts Personal Identifiable Information (PII) in code."""
import logging

import dask.dataframe as dd

from fondant.component import DaskTransformComponent
from fondant.component_spec import ComponentSpec
from fondant.logger import configure_logging
from fondant.manifest import Manifest
import fasttext

configure_logging()
logger = logging.getLogger(__name__)


class LanguageIdentification:
    """Language identification wrapper"""
    def __init__(self):
        pretrained_lang_model_weight_path = "./lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model_weight_path)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1)
        return predictions

    def is_language(self, row, language):
        return language in self.predict_lang(row["text"])[0][0]

class LanguageFilterComponent(DaskTransformComponent):
    """Component that detects and redacts PII from code."""

    def transform(
            self,
            *,
            dataframe: dd.DataFrame,
            language: str,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe.

        Returns:
            Dask dataframe
        """


        # Create a boolean mask using the custom function
        lang_detector = LanguageIdentification()
        mask = dataframe.map_partitions(lambda df: df.apply(lambda row: lang_detector.is_language(row, language), axis=1),
                                        meta=bool)

        # Filter the DataFrame using the mask
        return dataframe[mask]


if __name__ == "__main__":
    # component = LanguageFilterComponent.from_args()
