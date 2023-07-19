"""This component deduplicates images based on embeddings.

As proposed in [Abbas et al., 2022](https://arxiv.org/abs/2303.09540).

Implementation is based on https://github.com/conceptofmind/SemDeDup/tree/main.
"""
import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from fondant.component import DaskTransformComponent
from fondant.executor import DaskTransformExecutor

logger = logging.getLogger(__name__)


def sort_by_centroid_distance(embeddings, centroid, descending=True):
    distances = cdist(embeddings, centroid.reshape(1, -1), "euclidean")
    sorted_indices = np.argsort(distances, axis=0)
    if descending:
        sorted_indices = sorted_indices[::-1]
    return embeddings[sorted_indices.flatten()]


def deduplicate(df, epsilon):
    embeddings = df.image_embedding
    cluster_embeddings = np.vstack(embeddings)

    # get cluster centroid
    cluster_center = np.mean(cluster_embeddings, axis=0)

    # sort the cluster embeddings by the distance to the cluster centroid
    cluster_embeddings = sort_by_centroid_distance(
        cluster_embeddings, cluster_center, descending=True
    )

    # compute the pairwise cosine similarity between embeddings
    pairwise_sim_matrix = cosine_similarity(cluster_embeddings)

    # get upper triangular part of the matrix (excluding the diagonal)
    triu_sim_matrix = np.triu(pairwise_sim_matrix, k=1)

    # find max value in each column
    M = np.max(triu_sim_matrix, axis=0)

    # Check if the maximum similarity <= the threshold.
    points_to_keep_from_cluster_i = M <= 1 - epsilon

    print(points_to_keep_from_cluster_i)

    # Convert to boolean Pandas series
    points_to_keep_from_cluster_i = pd.Series(
        points_to_keep_from_cluster_i, index=embeddings.index
    )

    # add the points to keep to the list
    index_to_keep = points_to_keep_from_cluster_i[
        points_to_keep_from_cluster_i == True
    ].index

    return list(index_to_keep)


class DedupImageEmbeddingsComponent(DaskTransformComponent):
    """Component that deduplicates images based on embeddings."""

    def __init__(self, *_, epsilon: float) -> None:
        self.epsilon = epsilon

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe

        Returns:
            Dask dataframe
        """

        # filter dataframe
        indices_to_keep = dataframe.groupby("image_cluster_label").apply(deduplicate,
                                                                         epsilon=self.epsilon,
                                                                         meta=pd.Series(dtype="int8"))

        indices_to_keep = list(indices_to_keep.explode())

        # filter dataframe
        dataframe = dataframe.loc[dataframe.index.isin(indices_to_keep)]

        return dataframe


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(DedupImageEmbeddingsComponent)