import logging
import typing as t

import dask.dataframe as dd
import pandas as pd
from fondant.component import DaskLoadComponent
from fondant.core.schema import Field

logger = logging.getLogger(__name__)


class CSVReader(DaskLoadComponent):
    def __init__(
        self,
        *,
        produces: t.Dict[str, Field],
        dataset_uri: str,
        column_separator: str,
        column_name_mapping: t.Optional[dict],
        n_rows_to_load: t.Optional[int],
        index_column: t.Optional[str],
        **kwargs,
    ) -> None:
        """
        Args:
            spec: the component spec
            produces: The schema the component should produce
            dataset_uri: The remote path to the csv file/folder containing the dataset
            column_separator: Separator to use when parsing csv
            column_name_mapping: Mapping of the consumed dataset to fondant column names
            n_rows_to_load: optional argument that defines the
                number of rows to load. Useful for testing pipeline
                runs on a small scale.
            index_column: Column to set index to in the load component,
                if not specified a default globally unique index will
                be set.
            kwargs: Unhandled keyword arguments passed in by Fondant.
        """
        self.dataset_uri = dataset_uri
        self.column_separator = column_separator
        self.column_name_mapping = column_name_mapping
        self.n_rows_to_load = n_rows_to_load
        self.index_column = index_column
        self.produces = produces

    def get_columns_to_keep(self) -> t.List[str]:
        # Only read required columns
        columns = []

        if self.column_name_mapping:
            invert_column_name_mapping = {
                v: k for k, v in self.column_name_mapping.items()
            }
        else:
            invert_column_name_mapping = {}

        for field_name, field in self.produces.items():
            column_name = field_name
            if invert_column_name_mapping and column_name in invert_column_name_mapping:
                columns.append(invert_column_name_mapping[column_name])
            else:
                columns.append(column_name)

        if self.index_column is not None:
            columns.append(self.index_column)

        return columns

    def set_df_index(self, dask_df: dd.DataFrame) -> dd.DataFrame:
        if self.index_column is None:
            logger.info(
                "Index column not specified, setting a globally unique index",
            )

            def _set_unique_index(dataframe: pd.DataFrame, partition_info=None):
                """
                Function that sets a unique index
                based on the partition and row number.
                """
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
                for field_name, field in self.produces.items():
                    meta_dict[field_name] = pd.Series(
                        dtype=pd.ArrowDtype(field.type.value),
                    )
                return pd.DataFrame(meta_dict).set_index("id")

            meta = _get_meta_df()
            dask_df = dask_df.map_partitions(_set_unique_index, meta=meta)
        else:
            logger.info(f"Setting `{self.index_column}` as index")
            dask_df = dask_df.set_index(self.index_column, drop=True)

        return dask_df

    def return_subset_of_df(self, dask_df: dd.DataFrame) -> dd.DataFrame:
        if self.n_rows_to_load is not None:
            partitions_length = 0
            npartitions = 1
            for npartitions, partition in enumerate(dask_df.partitions, start=1):
                if partitions_length >= self.n_rows_to_load:
                    logger.info(
                        f"""Required number of partitions to load\n
                    {self.n_rows_to_load} is {npartitions}""",
                    )
                    break
                partitions_length += len(partition)
            dask_df = dask_df.head(self.n_rows_to_load, npartitions=npartitions)
            dask_df = dd.from_pandas(dask_df, npartitions=npartitions)
        return dask_df

    def load(self) -> dd.DataFrame:
        # 1) Load data, read as Dask dataframe
        logger.info("Loading dataset from the hub...")

        columns = self.get_columns_to_keep()

        logger.debug(f"Columns to keep: {columns}")
        dask_df = dd.read_csv(
            self.dataset_uri,
            sep=self.column_separator,
            usecols=columns,
        )

        # 2) Rename columns
        if self.column_name_mapping:
            logger.info("Renaming columns...")
            dask_df = dask_df.rename(columns=self.column_name_mapping)

        # 4) Optional: only return specific amount of rows
        dask_df = self.return_subset_of_df(dask_df)

        # 5) Set the index
        dask_df = self.set_df_index(dask_df)
        return dask_df
