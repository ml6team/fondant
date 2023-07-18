"""This component applies k-means clustering on image embeddings.
"""
import logging

import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.cluster import KMeans

from fondant.component import DaskTransformComponent
from fondant.executor import DaskTransformExecutor

logger = logging.getLogger(__name__)


class ClusterImageEmbeddingsComponent(DaskTransformComponent):
    """Component that clusters images based on embeddings."""

    def __init__(self, sample_ratio: float, num_clusters: int) -> None:
        self.sample_ratio = sample_ratio
        self.num_clusters = num_clusters

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        embeddings = dataframe["image_embedding"].sample(
            frac=self.sample_ratio, random_state=1
        )
        embeddings = np.vstack(list(embeddings))

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto")
        kmeans = kmeans.fit(embeddings)

        # call predict per row
        # # TODO call per partition?
        dataframe["image_cluster_label"] = dataframe["image_embedding"].apply(
            lambda example: kmeans.predict(example.reshape(1, -1))[0],
            meta=pd.Series(dtype="int"),
        )

        return dataframe


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(ClusterImageEmbeddingsComponent)
