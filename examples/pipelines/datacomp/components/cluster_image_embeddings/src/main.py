"""This component applies k-means clustering on image embeddings.
"""
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from fondant.component import DaskTransformComponent

logger = logging.getLogger(__name__)


class ClusterImageEmbeddingsComponent(DaskTransformComponent):
    """Component that clusters images based on embeddings."""

    def transform(
        self, dataframe: pd.DataFrame, num_points: int, num_clusters: int
    ) -> pd.DataFrame:
        """
        Args:
            dataframe: Pandas dataframe

        Returns:
            Pandas dataframe
        """
        embeddings = dataframe["image"]["embedding"].sample(
            n=num_points, random_state=1
        )
        embeddings = np.vstack(list(embeddings))

        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        kmeans = kmeans.fit(embeddings)

        cluster_labels = kmeans.predict(embeddings)

        # add column
        return pd.Series(cluster_labels).to_frame(name=("image", "cluster_label"))


if __name__ == "__main__":
    component = ClusterImageEmbeddingsComponent.from_args()
    component.run()
