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
from langchain.text_splitter import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    Language,
    LatexTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    NLTKTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

logger = logging.getLogger(__name__)

SUPPORTED_CHUNK_STRATEGIES = [
    "RecursiveCharacterTextSplitter",
    "HTMLHeaderTextSplitter",
    "CharacterTextSplitter",
    "Language",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "SentenceTransformersTokenTextSplitter",
    "LatexTextSplitter",
    "SpacyTextSplitter",
    "TokenTextSplitter",
    "NLTKTextSplitter",
    "PythonCodeTextSplitter",
    "character",
    "NLTK",
    "SpaCy",
]


class ChunkTextComponent(PandasTransformComponent):
    """Component that chunks text into smaller segments.
    More information about the different chunking strategies can be here:
      - https://python.langchain.com/docs/modules/data_connection/document_transformers/
      - https://www.pinecone.io/learn/chunking-strategies/.
    """

    def __init__(
        self,
        *,
        chunk_strategy: t.Optional[str],
        chunk_kwargs: t.Optional[dict],
        language_text_splitter: t.Optional[str],
        **kwargs,
    ):
        """
        Args:
            chunk_strategy: The strategy to use for chunking. One of
            ['RecursiveCharacterTextSplitter', 'HTMLHeaderTextSplitter', 'CharacterTextSplitter',
            'Language', 'MarkdownHeaderTextSplitter', 'MarkdownTextSplitter',
            'SentenceTransformersTokenTextSplitter', 'LatexTextSplitter', 'SpacyTextSplitter',
            'TokenTextSplitter', 'NLTKTextSplitter', 'PythonCodeTextSplitter', 'character',
            'NLTK', 'SpaCy']
            chunk_kwargs: Keyword arguments to pass to the chunker class.
            language_text_splitter: The programming language to use for splitting text into
            sentences if "language" is selected as the splitter. Check
            https://python.langchain.com/docs/modules/data_connection/document_transformers/
            code_splitter
            for more information on supported languages.
            kwargs: Unhandled keyword arguments passed in by Fondant.
        """
        self.chunk_strategy = chunk_strategy
        self.chunk_kwargs = chunk_kwargs
        self.chunker = self._get_chunker_class(chunk_strategy)
        self.language_text_splitter = language_text_splitter

    def _get_chunker_class(self, chunk_strategy: t.Optional[str]) -> TextSplitter:
        """
        Function to retrieve chunker class by string
        Args:
            chunk_strategy: The strategy to use for chunking. One of
            ['RecursiveCharacterTextSplitter', 'HTMLHeaderTextSplitter', 'CharacterTextSplitter',
            'Language', 'MarkdownHeaderTextSplitter', 'MarkdownTextSplitter',
            'SentenceTransformersTokenTextSplitter', 'LatexTextSplitter', 'SpacyTextSplitter',
            'TokenTextSplitter', 'NLTKTextSplitter', 'PythonCodeTextSplitter', 'character',
            'NLTK', 'SpaCy', 'recursive'].
        """
        class_dict = {
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
            "HTMLHeaderTextSplitter": HTMLHeaderTextSplitter,
            "CharacterTextSplitter": CharacterTextSplitter,
            "Language": Language,
            "MarkdownHeaderTextSplitter": MarkdownHeaderTextSplitter,
            "MarkdownTextSplitter": MarkdownTextSplitter,
            "SentenceTransformersTokenTextSplitter": SentenceTransformersTokenTextSplitter,
            "LatexTextSplitter": LatexTextSplitter,
            "SpacyTextSplitter": SpacyTextSplitter,
            "TokenTextSplitter": TokenTextSplitter,
            "NLTKTextSplitter": NLTKTextSplitter,
            "PythonCodeTextSplitter": PythonCodeTextSplitter,
        }

        if chunk_strategy not in SUPPORTED_CHUNK_STRATEGIES:
            msg = f"Chunk strategy must be one of: {SUPPORTED_CHUNK_STRATEGIES}"
            raise ValueError(
                msg,
            )

        if chunk_strategy == "Language":
            supported_languages = [e.value for e in Language]

            if self.language_text_splitter is None:
                msg = (
                    f"Language text splitter must be specified when using Language"
                    f" chunking strategy, choose from: {supported_languages}"
                )
                raise ValueError(
                    msg,
                )

            if self.language_text_splitter not in supported_languages:
                msg = f"Language text splitter must be one of: {supported_languages}"
                raise ValueError(
                    msg,
                )

            return RecursiveCharacterTextSplitter.from_language(
                language=Language(self.language_text_splitter),
                **self.chunk_kwargs,
            )

        return class_dict[chunk_strategy](**self.chunk_kwargs)

    def chunk_text(self, row) -> t.List[t.Tuple]:
        # Multi-index df has id under the name attribute
        doc_id = row.name
        text_data = row["text"]
        docs = self.chunker.create_documents([text_data])

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
