import logging
from surt import surt
import dask.dataframe as dd
import pandas as pd

from fondant.component import DaskTransformComponent

logger = logging.getLogger(__name__)


def get_license_strictness(license_string):
    """Returns strictness of the license keys."""
    strictness_order = {
        "by": 1,
        "by-sa": 2,
        "by-nd": 3,
        "by-nc": 4,
        "by-nc-sa": 5,
        "by-nc-nd": 6,
    }

    return strictness_order.get(license_string, float("inf"))


def local_dedup(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """Local dataframe deduplication, keep the first entry (most strict one)"""
    dataframe = dataframe.sort_index().sort_values(
        by=["image_license_type"],
        key=lambda col: col.apply(get_license_strictness),
    )

    return dataframe.drop_duplicates(subset=[column], keep="first")


class ImageUrlDeduplication(DaskTransformComponent):
    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """Deduplicate images based on source urls
        Args:
            df: A dask dataframe with the image urls
        Returns:
            A dask dataframe with the image urls
        """
        num_partitions = dataframe.npartitions

        # Create url_surtkey of image url
        dataframe["image_image_surt_url"] = dataframe["image_image_url"].apply(
            lambda x: surt(x), meta=("str")
        )

        dataframe = dataframe.set_index(
            "image_image_surt_url", npartitions=num_partitions
        )

        # Sort the DataFrame on the column
        return dataframe.map_partitions(local_dedup, "image_image_url")
