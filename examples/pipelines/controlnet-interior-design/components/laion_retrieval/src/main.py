"""
This component retrieves image URLs from LAION-5B based on a set of seed prompts.
"""
import logging
from typing import List

import dask.dataframe as dd
import dask.array as da

from clip_client import ClipClient, Modality

from fondant.component import FondantTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def query_clip_client(text: str, client: ClipClient) -> List[str]:
    """
    Given a text query and a ClipClient instance, this function retrieves
    image URLs and their LAION ids related to the query.

    Args:
        text: the text query
        client (ClipClient): an instance of ClipClient used to query the images

    Returns:
        urls: a list of strings, each representing a URL of an image related to the query
        ids: a list of integers, each representing an id in the LAION-5B dataset
    """
    results = client.query(text=text)
    urls = [i["url"] for i in results]
    ids = [i["id"] for i in results]

    return urls, ids


class LAIONRetrievalComponent(FondantTransformComponent):
    """
    Component that retrieves image URLs from LAION-5B based on a set of prompts.
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
        *,
        num_images: int,
        aesthetic_score: int,
        aesthetic_weight: float
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            num_images: number of images to retrieve for each prompt
            aesthetic_score: ranking score for aesthetic embedding, higher is prettier, between 0 and 9.
            aesthetic_weight: weight of the aesthetic embedding to add to the query, between 0 and 1.

        Returns:
            Dask dataframe
        """
        client = ClipClient(
            url="https://knn.laion.ai/knn-service",
            indice_name="laion5B-L-14",
            num_images=num_images,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
            modality=Modality.IMAGE,
        )

        # dataframe["images_url"] = dataframe["prompts_text"].apply(
        #     lambda example: query_clip_client(example, client),
        #     meta=("images_url", "str"),
        # )

        dataframe = dataframe["prompts_text"].apply(
            lambda example: query_clip_client(text=example, client=client),
            axis=1,
            result_type="expand",
            meta={0: str, 1: int},
        )
        dataframe.columns = ["images_url", "id"]

        # unpack list of urls and ids
        dataframe = dataframe.explode("images_url")
        dataframe = dataframe.explode("id")

        # add source column
        dataframe["source"] = "laion"

        # reorder columns
        dataframe = dataframe[["id", "source", "images_url"]]

        return dataframe


if __name__ == "__main__":
    component = LAIONRetrievalComponent.from_file()
    component.run()
