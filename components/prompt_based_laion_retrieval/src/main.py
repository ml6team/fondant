"""
This component retrieves image URLs from LAION-5B based on a set of seed prompts.
"""
import logging
from requests.exceptions import ConnectionError
from typing import List

import dask.dataframe as dd
import dask.array as da

from clip_client import ClipClient, Modality

from fondant.component import TransformComponent
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
    """
    results = client.query(text=text)
    try:
        urls = [i["url"] for i in results]
    except ConnectionError as e:
        urls = ["" for _ in range(len(results))]

    return urls


class LAIONRetrievalComponent(TransformComponent):
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

        logger.info("Retrieving URLs...")
        dataframe["images_url"] = dataframe["prompts_text"].apply(
            lambda example: query_clip_client(example, client),
            meta=("images_url", "str"),
        )

        # unpack list of urls
        dataframe = dataframe.explode("images_url")

        # add id and source columns
        # TODO use LAION id instead
        dataframe["id"] = dataframe.assign(id=1).id.cumsum()
        dataframe["source"] = "laion"

        # reorder columns
        dataframe = dataframe[["id", "source", "images_url"]]

        dataframe = dataframe.astype({'id': 'string', 'source': 'string'})

        dataframe = dataframe.reset_index(drop=True)

        return dataframe


if __name__ == "__main__":
    component = LAIONRetrievalComponent.from_file()
    component.run()
