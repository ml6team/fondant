"""A component that generates minhashes of text."""
import logging

import numpy as np
import pandas as pd
from datasketch import MinHash
from fondant.component import PandasTransformComponent
from nltk.util import ngrams

logger = logging.getLogger(__name__)


def create_shingles(text: str) -> list:
    """Creates text shingles that will be used for the hash generation."""
    # Split text into words
    words = text.split()

    # Generate shingles of size 3 using nltk's ngrams function
    return list(ngrams(words, 3))


def compute_minhash(shingles: list) -> np.ndarray:
    """Calculate minhash based on the shingles."""
    minhash = MinHash()

    # Update the MinHash object with the shingles
    for shingle in shingles:
        minhash.update(" ".join(shingle).encode("utf-8"))

    return minhash.hashvalues


class MinHashGeneratorComponent(PandasTransformComponent):
    """Component generates minhashes of text."""

    def __init__(self, *_, shingle_ngram_size: int):
        """Setup component.

        Args:
            shingle_ngram_size: Defines size of ngram used for the shingle generation.
        """
        self.shingle_ngram_size = shingle_ngram_size

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generates minhash values of text.

        Args:
            dataframe: Pandas dataframe.

        Returns:
            Pandas dataframe
        """
        dataframe[("text", "shingles")] = dataframe[("text", "data")].apply(
            create_shingles,
        )
        dataframe[("text", "minhash")] = dataframe[("text", "shingles")].apply(
            compute_minhash,
        )

        return dataframe
