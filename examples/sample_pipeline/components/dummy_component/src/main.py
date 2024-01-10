"""
Component that chunks text into smaller segments.

This component takes a body of text and chunks into small chunks. The id of the returned dataset
consists of the id of the original document followed by the chunk index.

"""
import logging

import pandas as pd
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class DummyComponent(PandasTransformComponent):
    """Dummy component that returns the dataframe as it is."""

    def __init__(self, *_, **kwargs):
        pass

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Dummy component that returns the dataframe as it is."""
        # raise RuntimeError
        return dataframe
