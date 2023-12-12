import logging
import subprocess
import sys
import typing as t
from collections import defaultdict

import dask.dataframe as dd
import pandas as pd
from fondant.component import DaskLoadComponent
from fondant.core.component_spec import ComponentSpec
from llama_index import download_loader

logger = logging.getLogger(__name__)


class LlamaHubReader(DaskLoadComponent):
    def __init__(
        self,
        spec: ComponentSpec,
        *,
        loader_class: str,
        loader_kwargs: dict,
        load_kwargs: dict,
        additional_requirements: t.List[str],
        n_rows_to_load: t.Optional[int] = None,
        index_column: t.Optional[str] = None,
    ) -> None:
        """
        Args:
            spec: the component spec
            loader_class: The name of the LlamaIndex loader class to use
            loader_kwargs: Keyword arguments to pass when instantiating the loader class
            load_kwargs: Keyword arguments to pass to the `.load()` method of the loader
            additional_requirements: Additional Python requirements to install
            n_rows_to_load: optional argument that defines the number of rows to load.
                Useful for testing pipeline runs on a small scale.
            index_column: Column to set index to in the load component, if not specified a default
                globally unique index will be set.
        """
        self.n_rows_to_load = n_rows_to_load
        self.index_column = index_column
        self.spec = spec

        self.install_additional_requirements(additional_requirements)

        loader_cls = download_loader(loader_class)
        self.loader = loader_cls(**loader_kwargs)
        self.load_kwargs = load_kwargs

    @staticmethod
    def install_additional_requirements(additional_requirements: t.List[str]):
        for requirement in additional_requirements:
            subprocess.check_call(  # nosec
                [sys.executable, "-m", "pip", "install", requirement],
            )

    def set_df_index(self, dask_df: dd.DataFrame) -> dd.DataFrame:
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
                for field_name, field in self.spec.produces.items():
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

    def load(self) -> dd.DataFrame:
        try:
            documents = self.loader.lazy_load_data(**self.load_kwargs)
        except NotImplementedError:
            documents = self.loader.load_data(**self.load_kwargs)

        doc_dict = defaultdict(list)
        for d, document in enumerate(documents):
            for column in self.spec.produces:
                if column == "text":
                    doc_dict["text"].append(document.text)
                else:
                    doc_dict[column].append(document.metadata.get(column))

            if d == self.n_rows_to_load:
                break

        dask_df = dd.from_dict(doc_dict, npartitions=1)

        dask_df = self.set_df_index(dask_df)
        return dask_df
