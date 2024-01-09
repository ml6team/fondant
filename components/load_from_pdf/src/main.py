import logging
import os
import tempfile
import typing as t
from collections import defaultdict

import dask.dataframe as dd
import fsspec as fs
import pandas as pd
from fondant.component import DaskLoadComponent
from fondant.core.component_spec import OperationSpec
from langchain_community.document_loaders import PyPDFDirectoryLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PDFReader(DaskLoadComponent):
    def __init__(
        self,
        spec: OperationSpec,
        *,
        pdf_path: str,
        n_rows_to_load: t.Optional[int] = None,
        index_column: t.Optional[str] = None,
    ) -> None:
        """
        Args:
            spec: the operation spec for the component
            pdf_path: Path to the PDF file
            n_rows_to_load: optional argument that defines the number of rows to load.
                Useful for testing pipeline runs on a small scale.
            index_column: Column to set index to in the load component, if not specified a default
                globally unique index will be set.
        """
        self.spec = spec
        self.pdf_path = pdf_path
        self.n_rows_to_load = n_rows_to_load
        self.index_column = index_column
        self.protocol = fs.utils.get_protocol(self.pdf_path)
        self.fs, _ = fs.core.url_to_fs(self.pdf_path)

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
                for field_name, field in self.spec.inner_produces.items():
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
        if self.protocol == "file":
            logger.info("Found PDF files local file system")

            if self.fs.exists(self.pdf_path):
                if self.fs.isdir(self.pdf_path):
                    pdf_dir = self.pdf_path
                else:
                    pdf_dir = os.path.dirname(self.pdf_path)

                loader = PyPDFDirectoryLoader(pdf_dir)
                documents = loader.load()

            else:
                msg = "PDF path does not exist"
                raise ValueError(msg)

        else:
            logger.info("Found PDF files on remote file system")

            files = self.fs.ls(self.pdf_path)

            with tempfile.TemporaryDirectory() as temp_dir:
                for file_path in tqdm(files):
                    if file_path.endswith(".pdf"):
                        file_name = os.path.basename(file_path)
                        temp_file_path = os.path.join(temp_dir, file_name)
                        self.fs.get(file_path, temp_file_path)

                loader = PyPDFDirectoryLoader(temp_dir)
                documents = loader.lazy_load()

        doc_dict = defaultdict(list)
        for doc_counter, document in enumerate(documents):
            doc_dict["file_name"].append(os.path.basename(document.metadata["source"]))
            doc_dict["text"].append(document.page_content)

            if doc_counter == self.n_rows_to_load:
                break

        dask_df = dd.from_dict(doc_dict, npartitions=1)

        dask_df = self.set_df_index(dask_df)
        return dask_df
