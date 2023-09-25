"""A component that downloads common crawl files."""
import logging
import mimetypes
import pandas as pd
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class FileTypeFilter(PandasTransformComponent):
    """Custom component to filter on specific file type based on url"""

    def __init__(self, *args, mime_type: str):
        self.mime_type = mime_type

    @staticmethod
    def get_mime_type(data):
        """Guess mime type based on the file name"""
        mime_type, _ = mimetypes.guess_type(data)
        return mime_type

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Reduce dataframe to specific mime type"""
        dataframe[("images", "mime_type")] = dataframe[("images", "url")].apply(
            self.get_mime_type
        )
        return dataframe[dataframe[("images", "mime_type")] == self.mime_type]
