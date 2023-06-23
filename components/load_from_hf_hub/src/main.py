"""This component loads a seed dataset from the hub."""
import logging
import typing as t

import dask.dataframe as dd

from fondant.component import LoadComponent

logger = logging.getLogger(__name__)


class LoadFromHubComponent(LoadComponent):
    def load(self,
             *,
             dataset_name: str,
             column_name_mapping: dict,
             image_column_names: t.Optional[list],
             n_rows_to_load: t.Optional[int]) -> dd.DataFrame:
        """
        Args:
            dataset_name: name of the dataset to load.
            column_name_mapping: Mapping of the consumed hub dataset to fondant column names
            image_column_names: A list containing the original hub image column names. Used to
                format the image from HF hub format to a byte string
            n_rows_to_load: optional argument that defines the number of rows to load. Useful for
              testing pipeline runs on a small scale
        Returns:
            Dataset: HF dataset.
        """
        # 1) Load data, read as Dask dataframe
        logger.info("Loading dataset from the hub...")
        dask_df = dd.read_parquet(f"hf://datasets/{dataset_name}")

        # 2) Make sure images are bytes instead of dicts
        if image_column_names is not None:
            for image_column_name in image_column_names:
                dask_df[image_column_name] = dask_df[image_column_name].map(
                    lambda x: x["bytes"], meta=("bytes", bytes),
                )

        # 3) Rename columns
        dask_df = dask_df.rename(columns=column_name_mapping)

        # 4) Optional: only return specific amount of rows

        if n_rows_to_load:
            dask_df = dask_df.head(n_rows_to_load)
            dask_df = dd.from_pandas(dask_df, npartitions=1)

        return dask_df


if __name__ == "__main__":
    component = LoadFromHubComponent.from_args()
    component.run()
