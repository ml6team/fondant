"""
This component retrieves image URLs from LAION-5B based on a set of seed prompts.
"""
import logging
from typing import List

import dask.dataframe as dd

from clip_retrieval.clip_client import ClipClient, Modality

from fondant.component import FondantTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def query_clip_client(text: str, client: ClipClient) -> List[str]:
    """
    Given a text query and a ClipClient instance, this function retrieves
    image URLs related to the query.

    Args:
        text: the text query
        client (ClipClient): an instance of ClipClient used to query the images

    Returns:
        results: a list of strings, each representing a URL of an image related to the query
    """
    results = client.query(text=text)
    results = [i["url"] for i in results]

    return results


class RetrieveImagesComponent(FondantTransformComponent):
    """
    Component that retrieves images from LAION-5B based on a set of prompts.
    """

    def transform(
        self,
        *,
        dataframe: dd.DataFrame,
        num_images: int,
        aesthetic_score: int,
        aesthetic_weight: float
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            num_images: number of images to retrieve for each prompt
            aesthetic_score: ranking score for aesthetic, higher is prettier.
            aesthetic_weight: weight of the aesthetic score, between 0 and 1.

        Returns:
            Dask dataframe
        """
        client = ClipClient(
            url="https://knn.laion.ai/knn-service",
            indice_name="laion5B-L-14",
            num_images=num_images,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=0.5,
            modality=Modality.IMAGE,
        )

        df["images_url"] = df["prompts_text"].map_partitions(
            lambda x: x.apply(query_clip_client, args=(client,)),
            meta=("images_url", "str"),
        )

        # unpack list of urls
        df = df.explode("image_urls")

        return df


if __name__ == "__main__":
    component = RetrieveImagesComponent.from_file("../fondant_component.yaml")
    component.run()
