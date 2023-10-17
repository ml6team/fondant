"""
Component that chunks text into smaller segments.

This component takes a body of text and chunks into small chunks. The id of the returned dataset
consists of the id of the original document followed by the chunk index.

"""
import itertools
import logging
import typing as t

import pandas as pd
from fondant.component import PandasTransformComponent
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_text(row, text_splitter: RecursiveCharacterTextSplitter) -> t.List[t.Tuple]:
    # Multi-index df has id under the name attribute
    doc_id = row.name
    text_data = row[("text", "data")]
    docs = text_splitter.create_documents([text_data])
    return [
        (f"{doc_id}_{chunk_id}", chunk.page_content)
        for chunk_id, chunk in enumerate(docs)
    ]


class DownloadImagesComponent(PandasTransformComponent):
    """Component that downloads images based on URLs."""

    def __init__(
        self,
        *_,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """
        Args:
            chunk_size: Maximum size of chunks to return.
            chunk_overlap: Overlap in characters between chunks.

        Returns:
            Dask dataframe.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Chunking {len(dataframe)} documents...")

        results = dataframe.apply(
            chunk_text,
            args=(self.text_splitter,),
            axis=1,
        ).to_list()

        # Flatten results
        results = list(itertools.chain.from_iterable(results))

        # Turn into dataframes
        results_df = pd.DataFrame(results, columns=["id", "data"])
        results_df = results_df.set_index("id")

        # Set multi-index column for the expected subset and field
        results_df.columns = [["text"], ["data"]]

        return results_df
