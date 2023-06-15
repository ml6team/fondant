"""This component loads a seed dataset from the hub."""
import logging
import typing as t

import dask.dataframe as dd

from fondant.component import LoadComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class LoadFromHubComponent(LoadComponent):
    def load(self, *, dataset_name: str, column_name_mapping: dict,
             image_column_name: t.Optional[str]) -> dd.DataFrame:
        """
        Args:
            dataset_name: name of the dataset to load.
            column_name_mapping: column to map the original column names of the input dataset to
            image_column_name: the name of a column containing images. Used to convert the columns
            to a proper format. The column name can be either the original column name or the subset
            specific name
        Returns:
            Dataset: HF dataset.
        """
        # 1) Load data, read as Dask dataframe
        logger.info("Loading dataset from the hub...")
        dask_df = dd.read_parquet(f"hf://datasets/{dataset_name}")

        # 2) Rename columns
        dask_df = dask_df.rename(columns=column_name_mapping)

        # 3) Make sure images are bytes instead of dicts
        if image_column_name:
            if image_column_name in column_name_mapping:
                image_column_name = column_name_mapping[image_column_name]
            dask_df[image_column_name] = dask_df[image_column_name].map(
                lambda x: x["bytes"], meta=("bytes", bytes)
            )

        return dask_df


if __name__ == "__main__":
    component = LoadFromHubComponent.from_args()
    component.run()
