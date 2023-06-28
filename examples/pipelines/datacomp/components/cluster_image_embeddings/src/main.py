"""This component applies k-means clustering on image embeddings.
"""
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class ClusterImageEmbeddingsComponent(PandasTransformComponent):
    """Component that clusters images based on embeddings."""

    def setup(self, *, num_clusters: int) -> None:
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters, random_state=0, batch_size=6, n_init="auto"
        )

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            dataframe: Pandas dataframe

        Returns:
            Pandas dataframe
        """
        embeddings = dataframe["image"]["embedding"]
        embeddings = np.vstack(list(embeddings))

        # run minibatch k-means
        kmeans = self.kmeans.partial_fit(embeddings)

        cluster_labels = kmeans.predict(embeddings)

        # add column
        return pd.Series(cluster_labels).to_frame(name=("image", "cluster_label"))


if __name__ == "__main__":
    component = ClusterImageEmbeddingsComponent.from_args()
    component.run()
