"""This component applies k-means clustering on minhashes."""
import logging

import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.cluster import KMeans

from fondant.component import DaskTransformComponent
from fondant.executor import DaskTransformExecutor

logger = logging.getLogger(__name__)


class ClusterMinHashComponent(DaskTransformComponent):
    """Component that clusters images based on embeddings."""

    def __init__(self, *_, sample_ratio: float, num_clusters: int) -> None:
        self.sample_ratio = sample_ratio
        self.num_clusters = num_clusters

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe

        Returns:
            Dask dataframe
        """
        embeddings = dataframe["webpage_minhash"].sample(
            frac=self.sample_ratio, random_state=1
        )
        embeddings = np.vstack(list(embeddings))

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto")
        kmeans = kmeans.fit(embeddings)

        dataframe["webpage_cluster_label"] = dataframe["webpage_minhash"].apply(
            lambda example: kmeans.predict(example.reshape(1, -1))[0],
            meta=pd.Series(dtype="int"),
        )

        return dataframe


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(ClusterMinHashComponent)
