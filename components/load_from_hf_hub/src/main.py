"""This component loads a seed dataset from the hub."""
import logging
import typing as t

import dask.dataframe as dd
from fondant.component import DaskLoadComponent
from fondant.executor import DaskLoadExecutor

logger = logging.getLogger(__name__)


class LoadFromHubComponent(DaskLoadComponent):

    def __init__(self, *_,
             dataset_name: str,
             column_name_mapping: dict,
             image_column_names: t.Optional[list],
             n_rows_to_load: t.Optional[int],
    ) -> None:
        """
        Args:
            dataset_name: name of the dataset to load.
            column_name_mapping: Mapping of the consumed hub dataset to fondant column names
            image_column_names: A list containing the original hub image column names. Used to
                format the image from HF hub format to a byte string
            n_rows_to_load: optional argument that defines the number of rows to load. Useful for
              testing pipeline runs on a small scale.
        """
        self.dataset_name = dataset_name
        self.column_name_mapping = column_name_mapping
        self.image_column_names = image_column_names
        self.n_rows_to_load = n_rows_to_load

    def load(self) -> dd.DataFrame:
        # 1) Load data, read as Dask dataframe
        logger.info("Loading dataset from the hub...")
        dask_df = dd.read_parquet(f"hf://datasets/{self.dataset_name}")

        # 2) Make sure images are bytes instead of dicts
        if self.image_column_names is not None:
            for image_column_name in self.image_column_names:
                dask_df[image_column_name] = dask_df[image_column_name].map(
                    lambda x: x["bytes"], meta=("bytes", bytes),
                )

        # 3) Rename columns
        logger.info("Renaming columns...")
        dask_df = dask_df.rename(columns=self.column_name_mapping)

        # 4) Optional: only return specific amount of rows
        if self.n_rows_to_load is not None:
            partitions_length = 0 
            for npartitions, partition in enumerate(dask_df.partitions):
                if partitions_length >= self.n_rows_to_load:
                    logger.info(f"Required number of partitions to load {self.n_rows_to_load} is {npartitions}")
                    break 
                partitions_length += len(partition)
            dask_df = dask_df.head(self.n_rows_to_load, npartitions=npartitions)
            dask_df = dd.from_pandas(dask_df, npartitions=npartitions)

        # Set monotonically increasing index
        logger.info("Setting the index...")
        dask_df["id"] = 1
        dask_df["id"] = dask_df.id.cumsum()
        dask_df = dask_df.set_index("id", sort=True)

        return dask_df


if __name__ == "__main__":
    executor = DaskLoadExecutor.from_args()
    executor.execute(LoadFromHubComponent)
