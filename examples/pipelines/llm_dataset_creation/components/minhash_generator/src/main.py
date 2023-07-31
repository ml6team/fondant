"""A component that generates minhashes of text."""
import logging
from datasketch import MinHash
from nltk.util import ngrams

import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor

logger = logging.getLogger(__name__)


class MinHashGeneratorComponent(PandasTransformComponent):
    """Component generates minhashes of text."""

    @staticmethod
    def create_shingles(text):
        """Creates text shingles that will be used for the hash generation."""
        # Split text into words
        words = text.split()

        # Generate shingles of size 3 using nltk's ngrams function
        shingles = list(ngrams(words, 3))

        return shingles

    @staticmethod
    def compute_minhash(shingles):
        """Calculate minhash based on the shingles"""
        minhash = MinHash()

        # Update the MinHash object with the shingles
        for shingle in shingles:
            minhash.update(" ".join(shingle).encode("utf-8"))

        return minhash.hashvalues

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generates minhash values of text

        Args:
            dataframe: Pandas dataframe.

        Returns:
            Pandas dataframe
        """
        dataframe[("webpage", "shingles")] = dataframe[("webpage", "html")].apply(
            self.create_shingles
        )
        dataframe[("webpage", "minhash")] = dataframe[("webpage", "shingles")].apply(
            self.compute_minhash
        )

        return dataframe


if __name__ == "__main__":
    executor = PandasTransformExecutor.from_args()
    executor.execute(MinHashGeneratorComponent)
