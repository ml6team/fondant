import logging
import os
import typing as t

import dask.dataframe as dd
import faiss
import fsspec
import pandas as pd
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class RetrieveFromFaissByEmbedding(PandasTransformComponent):
    """Retrieve images from a faiss index using CLIP embeddings."""

    def __init__(  # PLR0913
        self,
        url_mapping_path: str,
        faiss_index_path: str,
        num_images: int = 2,
    ):
        self.number_of_images = num_images

        # Download faiss index to local machine
        if not os.path.exists("faiss_index"):
            logger.info(f"Downloading faiss index from {faiss_index_path}")
            with fsspec.open(faiss_index_path, "rb") as f:
                file_contents = f.read()

            with open("faiss_index", "wb") as out:
                out.write(file_contents)

        dataset = dd.read_parquet(url_mapping_path)
        if "url" not in dataset.columns:
            msg = "Dataset does not contain column 'url'"
            raise ValueError(msg)
        self.image_urls = dataset["url"].compute().to_list()

    def retrieve_from_index(
        self,
        query: float,
        number_of_images: int = 2,
    ) -> t.List[str]:
        """Retrieve images from faiss index."""
        search_index = faiss.read_index("faiss_index")
        _, indices = search_index.search(query, number_of_images)
        return indices.tolist()[0]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Transform partition of dataframe."""
        results = []
        for index, row in dataframe.iterrows():
            embedding = row["embedding"]

            indices = self.retrieve_from_index(embedding, self.number_of_images)
            for i, idx in enumerate(indices):
                url = self.image_urls[idx]
                results.append((index, url))

        results_df = pd.DataFrame(
            results,
            columns=["id", "image_url"],
        )
        results_df = results_df.set_index("id")
        return results_df
