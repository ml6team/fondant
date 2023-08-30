import logging

import dask.dataframe as dd
import pandas as pd

from fondant.component import DaskTransformComponent

logger = logging.getLogger(__name__)


def get_license_strictness(license_string):
    """Returns strictness of the license keys."""
    strictness_order = {
        "by": 1,
        "by-sa": 2,
        "by-nc": 3,
        "by-nc-sa": 4,
        "by-nd": 5,
        "by-nc-nd": 6,
    }

    return strictness_order.get(license_string, 0)


def local_dedup(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """Local dataframe deduplication, keep the last entry (most strict one)"""
    return dataframe.drop_duplicates(subset=[column], keep="last")


class ImageUrlDeduplication(DaskTransformComponent):
    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """Deduplicate images based on source urls
        Args:
            df: A dask dataframe with the image urls
        Returns:
            A dask dataframe with the image urls
        """
        num_partitions = dataframe.npartitions

        # Sort the DataFrame on the column
        dataframe = dataframe.sort_values(
            by=["image_image_url", "image_license_type"],
            key=lambda col: col.apply(get_license_strictness),
        )

        # Repartition on column
        dataframe = dataframe.repartition(npartitions=num_partitions)

        return dataframe.map_partitions(local_dedup, "image_image_url")
