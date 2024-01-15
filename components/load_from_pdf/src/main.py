import logging
import os
import typing as t

import dask.dataframe as dd
import fitz
import fsspec as fs
import pandas as pd
from fondant.component import DaskLoadComponent
from fondant.core.schema import Field

logger = logging.getLogger(__name__)


class PDFReader(DaskLoadComponent):
    def __init__(
        self,
        produces: t.Dict[str, Field],
        *,
        pdf_path: str,
        n_rows_to_load: t.Optional[int] = None,
        index_column: t.Optional[str] = None,
        n_partitions: t.Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            produces: The schema the component should produce
            pdf_path: Path to the PDF file
            n_rows_to_load: optional argument that defines the number of rows to load.
                Useful for testing pipeline runs on a small scale.
            index_column: Column to set index to in the load component, if not specified a default
                globally unique index will be set.
            n_partitions: Number of partitions of the dask dataframe. If not specified, the number
                of partitions will be equal to the number of CPU cores. Set to high values if
                the data is large and the pipeline is running out of memory.
            kwargs: Unhandled keyword arguments passed in by Fondant.
        """
        self.produces = produces
        self.pdf_path = pdf_path
        self.n_rows_to_load = n_rows_to_load
        self.index_column = index_column
        self.protocol = fs.utils.get_protocol(self.pdf_path)
        self.fs, _ = fs.core.url_to_fs(self.pdf_path)
        self.n_partitions = n_partitions if n_partitions is not None else os.cpu_count()

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

    def load_pdf_from_fs(self, file_path: str):
        with self.fs.open(file_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        documents = fitz.open("pdf", pdf_bytes)
        # get all text
        text = "".join([document.get_text() for document in documents])
        documents.close()

        return text

    def process_pdf(self, row):
        file_path = row["pdf_path"]
        text = self.load_pdf_from_fs(file_path)
        row["file_name"] = file_path.split("/")[-1]  # Extracting filename
        row["text"] = text
        return row

    def load(self) -> dd.DataFrame:
        try:
            file_paths = self.fs.ls(self.pdf_path)
        except NotADirectoryError:
            file_paths = [self.pdf_path]

        file_paths = [
            file_path for file_path in file_paths if file_path.endswith(".pdf")
        ]

        if self.n_rows_to_load is not None:
            file_paths = file_paths[: self.n_rows_to_load]

        dask_df = dd.from_pandas(
            pd.DataFrame({"pdf_path": file_paths}),
            npartitions=self.n_partitions,
        )

        meta_dict = {}
        for field_name, field in self.produces.items():
            meta_dict[field_name] = pd.Series(
                dtype=pd.ArrowDtype(field.type.value),
            )
        meta_dict = pd.DataFrame(meta_dict)

        dask_df = dask_df.map_partitions(
            lambda part: part.apply(self.process_pdf, axis=1),
            meta=meta_dict,
        )

        dask_df = self.set_df_index(dask_df)
        return dask_df
