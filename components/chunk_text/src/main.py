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


class ChunkTextComponent(PandasTransformComponent):
    """Component that chunks text into smaller segments.."""

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        **kwargs,
    ):
        """
        Args:
            chunk_size: Maximum size of chunks to return.
            chunk_overlap: Overlap in characters between chunks.
            kwargs: Unhandled keyword arguments passed in by Fondant.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_text(self, row) -> t.List[t.Tuple]:
        # Multi-index df has id under the name attribute
        doc_id = row.name
        text_data = row["text"]
        docs = self.text_splitter.create_documents([text_data])
        return [
            (doc_id, f"{doc_id}_{chunk_id}", chunk.page_content)
            for chunk_id, chunk in enumerate(docs)
        ]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Chunking {len(dataframe)} documents...")

        results = dataframe.apply(
            self.chunk_text,
            axis=1,
        ).to_list()

        # Flatten results
        results = list(itertools.chain.from_iterable(results))

        # Turn into dataframes
        results_df = pd.DataFrame(
            results,
            columns=["original_document_id", "id", "text"],
        )
        results_df = results_df.set_index("id")

        return results_df
