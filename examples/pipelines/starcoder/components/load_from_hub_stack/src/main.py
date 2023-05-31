"""
This component loads a code dataset from a remote location on the Hugging Face hub.
"""

import logging

import dask.dataframe as dd

from fondant.component import LoadComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class LoadFromHubComponent(LoadComponent):
    def load(self, *, dataset_name: str) -> dd.DataFrame:
        """
        Args:
            dataset_name: name of the dataset to load

        Returns:
            Dataset: HF dataset
        """

        # 1) Load data, read as Dask dataframe
        logger.info("Loading dataset from the hub...")
        dask_df = dd.read_parquet(f"hf://datasets/{dataset_name}")

        # 2) Add prefix to column
        column_dict = {column: f"code_{column}" for column in dask_df.columns}
        dask_df = dask_df.rename(columns=column_dict)

        return dask_df


if __name__ == "__main__":
    component = LoadFromHubComponent.from_file()
    component.run()
