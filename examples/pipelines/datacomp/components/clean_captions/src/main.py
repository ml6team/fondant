import logging

import pandas as pd

from fondant.component import PandasTransformComponent
from dateutil.parser import parse

logger = logging.getLogger(__name__)


def isNonEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return True
    else:
        return False

def get_num_nonenglish_characters(text):
    return sum([isNonEnglish(char) for char in text])

def has_too_much_weird_characters(text, max_ratio=0.5):
    return (get_num_nonenglish_characters(text) / len(text)) > max_ratio

def is_valid_date(date_string):
    try:
        parse(date_string)
        return True
    except (ValueError, OverflowError):
        return False
    
def is_empty(text):
    return text.strip() != ""


class FilterTextComplexity(PandasTransformComponent):
    """Component that filters out bad captions in image-text pairs:
    - Empty captions
    - Captions with weird characters
    - Captions that are dates
    """

    def __init__(
        self,
        *args,
    ) -> None:
        pass

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        texts = dataframe["text"]["data"]

        logger.info("Filtering on empty captions...")
        mask = texts.apply(lambda text: not is_empty(text))
        dataframe = dataframe[mask]

        logger.info("Filtering on weird character captions...")
        mask = texts.apply(lambda text: not has_too_much_weird_characters(text))
        dataframe = dataframe[mask]

        logger.info("Filtering on captions that look like dates...")
        mask = texts.apply(lambda text: not is_valid_date(text))
        dataframe = dataframe[mask]

        return dataframe

