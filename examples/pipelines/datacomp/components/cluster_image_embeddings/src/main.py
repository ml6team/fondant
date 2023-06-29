"""This component applies k-means clustering on image embeddings.
"""
import logging

import dask.dataframe as dd
import numpy as np
from sklearn.cluster import KMeans

from fondant.component import DaskTransformComponent

logger = logging.getLogger(__name__)


class ClusterImageEmbeddingsComponent(DaskTransformComponent):
    """Component that clusters images based on embeddings."""

    def transform(
        self, dataframe: dd.DataFrame, sample_ratio: float, num_clusters: int
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe

        Returns:
            Dask dataframe
        """
        num_rows = dataframe.shape[0].compute()
        num_points = int(num_rows * sample_ratio)
        embeddings = dataframe["image_embedding"].sample(n=num_points, random_state=1)
        embeddings = np.vstack(list(embeddings))

        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        kmeans = kmeans.fit(embeddings)

        # add column
        dataframe["image_cluster_label"] = dataframe["image_embedding"].map_partitions(
            lambda example: kmeans.predict()
        )

        return dataframe


if __name__ == "__main__":
    component = ClusterImageEmbeddingsComponent.from_args()
    component.run()
