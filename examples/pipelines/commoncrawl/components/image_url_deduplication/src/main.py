import logging

import dask.dataframe as dd

from fondant.component import DaskTransformComponent

logger = logging.getLogger(__name__)


def deduplicate_dask_dataframe(dataframe: dd.DataFrame, column: str):
    logger.info(f"Start exact deduplication of column {column}.")

    num_partitions = dataframe.npartitions

    # Sort the DataFrame on the column
    dataframe = dataframe.sort_values(column)

    # Set index to column
    dataframe = dataframe.set_index(column)

    # Repartition on column
    dataframe = dataframe.repartition(npartitions=num_partitions)

    # Dedup partition wise on exact match
    return dataframe.index.unique()


class ImageUrlDeduplication(DaskTransformComponent):
    def __init__(self):
        pass

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """Deduplicate images based on source urls
        Args:
            df: A dask dataframe with the image urls
        Returns:
            A dask dataframe with the image urls
        """

        return deduplicate_dask_dataframe(dataframe, column="image_image_url")
