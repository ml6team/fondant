"""This component loads a seed dataset from the hub."""
import logging
import typing as t

import dask
import dask.dataframe as dd
import pandas as pd
from fondant.component import DaskLoadComponent
from fondant.component_spec import ComponentSpec

logger = logging.getLogger(__name__)

dask.config.set({"dataframe.convert-string": False})


class LoadFromParquet(DaskLoadComponent):

    def __init__(self,
                 spec: ComponentSpec,
                 *_,
                 dataset_uri: str,
                 column_name_mapping: dict,
                 n_rows_to_load: int,
                 index_column: t.Optional[str],
                 ) -> None:
        """
        Args:
            spec: the component spec
            dataset_uri: The remote path to the parquet file/folder containing the dataset
            column_name_mapping: Mapping of the consumed dataset to fondant column names
            n_rows_to_load: optional argument that defines the number of rows to load. Useful for
              testing pipeline runs on a small scale.
            index_column: Column to set index to in the load component, if not specified a default
                globally unique index will be set.
        """
        self.dataset_uri = dataset_uri
        self.column_name_mapping = column_name_mapping
        self.n_rows_to_load = n_rows_to_load
        self.index_column = index_column
        self.spec = spec

    def load(self) -> dd.DataFrame:
        # 1) Load data, read as Dask dataframe
        logger.info("Loading dataset from the file...")
        dask_df = dd.read_parquet(self.dataset_uri)

        # 2) Rename columns
        if self.column_name_mapping:
            logger.info("Renaming columns...")
            dask_df = dask_df.rename(columns=self.column_name_mapping)

        # 3) Optional: only return specific amount of rows
        if self.n_rows_to_load > 0:
            partitions_length = 0
            npartitions = 1
            for npartitions, partition in enumerate(dask_df.partitions, start=1):
                if partitions_length >= self.n_rows_to_load:
                    logger.info(f"""Required number of partitions to load\n
                    {self.n_rows_to_load} is {npartitions}""")
                    break
                partitions_length += len(partition)
            dask_df = dask_df.head(self.n_rows_to_load, npartitions=npartitions)
            dask_df = dd.from_pandas(dask_df, npartitions=npartitions)

        # 4) Set the index
        if self.index_column is None:
            logger.info(
                "Index column not specified, setting a globally unique index",
            )

            def _set_unique_index(dataframe: pd.DataFrame, partition_info=None):
                """Function that sets a unique index based on the partition and row number."""
                dataframe["id"] = 1
                dataframe["id"] = (
                        str(partition_info["number"])
                        + "_"
                        + (dataframe.id.cumsum()).astype(str)
                )
                dataframe.index = dataframe.pop("id")
                return dataframe

            def _get_meta_df() -> pd.DataFrame:
                meta_dict = {"id": pd.Series(dtype="object")}
                for subset_name, subset in self.spec.produces.items():
                    for field_name, field in subset.fields.items():
                        meta_dict[f"{subset_name}_{field_name}"] = pd.Series(
                            dtype=pd.ArrowDtype(field.type.value),
                        )
                return pd.DataFrame(meta_dict).set_index("id")

            meta = _get_meta_df()
            dask_df = dask_df.map_partitions(_set_unique_index, meta=meta)
        else:
            logger.info(f"Setting `{self.index_column}` as index")
            dask_df = dask_df.set_index(self.index_column, drop=True)

        return dask_df
