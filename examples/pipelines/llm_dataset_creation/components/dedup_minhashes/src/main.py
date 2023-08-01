"""This component deduplicates text based minhashes."""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
from fondant.component import DaskTransformComponent
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHashLSH, MinHash
from fondant.executor import DaskTransformExecutor

logger = logging.getLogger(__name__)


def deduplicate(df, epsilon, num_perm):
    minhashes = df["webpage_minhash"]

    lsh = MinHashLSH(threshold=epsilon, num_perm=num_perm)

    # Insert MinHash signatures into the LSH index
    for index, minhash in minhashes.items():
        _minhash = MinHash(hashvalues=minhash)
        try:
            lsh.insert(index, _minhash)
        except:
            # TODO this happens if indexes are duplicated accross partitions
            print("Key ", index, "already exists. Skipped.")

    # Find duplicates based on similarity threshold
    duplicated_indices = set()
    for _, minhash in minhashes.items():
        _minhash = MinHash(hashvalues=minhash)
        similar_docs = lsh.query(_minhash)
        if len(similar_docs) > 1:  # Duplicate documents found
            duplicated_entries = set(similar_docs[1:])
            duplicated_indices.update(duplicated_entries)

    # index offset is needed for partitiones - first element not always index=0
    index_offset = df.index[0]
    duplicated_indices = [
        int(el) - int(index_offset) for el in list(duplicated_indices)
    ]

    mask_array = np.zeros(len(df), dtype=bool)

    mask_array[duplicated_indices] = True

    points_to_keep_from_cluster_i = ~mask_array

    return df[points_to_keep_from_cluster_i].index.tolist()


class DedupMinHashesComponent(DaskTransformComponent):
    """Component that deduplicates minhashes with LSH."""

    def __init__(self, *_, epsilon: float, num_perm: int) -> None:
        self.epsilon = epsilon
        self.num_perm = num_perm

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            epsilon: Epsilon to use as threshold for jaccard similarity.

        Returns:
            Dask dataframe
        """
        indices_to_keep = dataframe.groupby("webpage_cluster_label").apply(
            deduplicate,
            epsilon=self.epsilon,
            num_perm=self.num_perm,
            meta=pd.Series(dtype=dataframe.index.dtype),
        )

        indices_to_keep = list(indices_to_keep.explode())

        # filter dataframe
        dataframe = dataframe.loc[dataframe.index.isin(indices_to_keep)]

        return dataframe


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(DedupMinHashesComponent)
