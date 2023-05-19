"""
This component retrieves image URLs from LAION-5B based on a set of seed prompts.
"""
import logging
from typing import List

import dask.dataframe as dd
import dask.array as da

from clip_client import ClipClient, Modality

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def query_clip_client(embedding_input, client: ClipClient) -> List[str]:
    """
    Given a text query and a ClipClient instance, this function retrieves
    image URLs and their LAION ids related to the query.

    Args:
        embedding_input (TODO): TODO
        client (ClipClient): an instance of ClipClient used to query the images

    Returns:
        urls: a list of strings, each representing a URL of an image related to the query
    """
    results = client.query(embedding_input=embedding_input.tolist())
    urls = [i["url"] for i in results]

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
        dataframe = dataframe.sample(frac=0.2)
        dataframe["images_url"] = dataframe["embeddings_data"].apply(
            lambda example: query_clip_client(example, client),
            meta=("images_url", "str"),
        )

        # # unpack list of urls
        dataframe = dataframe.explode("images_url")

        # drop NaN rows for images_url column
        dataframe = dataframe.dropna(subset=["images_url"])

        # add id and source columns
        dataframe["id"] = dataframe.assign(id=1).id.cumsum()
        dataframe["source"] = "laion"
        dataframe = dataframe.astype({'id': 'string', 'source': 'string', 'images_url': 'string'})

        # reorder columns
        dataframe = dataframe[["id", "source", "images_url"]]
        dataframe = dataframe.reset_index(drop=True)
        print(type(dataframe))
        # uid index is id+"_"+source
        dataframe["uid"] = dataframe["id"] + "_" + dataframe["source"]
        # set as index
        dataframe = dataframe.set_index("uid")
        return dataframe


if __name__ == "__main__":
    component = LAIONRetrievalComponent.from_file()
    component.run()