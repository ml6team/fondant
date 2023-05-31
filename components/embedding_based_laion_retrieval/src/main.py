"""
This component retrieves image URLs from LAION-5B based on a set of seed prompts.
"""
import asyncio
import concurrent.futures
import logging
import typing as t

import dask.dataframe as dd
import pandas as pd

from clip_client import ClipClient, Modality

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def query_clip_client(embeddings: pd.Series, client: ClipClient) -> pd.DataFrame:
    """
    Asynchronously retrieve image URLs and ids based on prompts in the provided dataframe.
    Args:
        embeddings: Series containing embedding
        client (ClipClient): an instance of ClipClient used to query the images
    Returns:
        urls: A dataframe with the image urls and the LAION ids as index
    """

    async def async_query():
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    client.query,
                    embedding
                )
                for embedding in embeddings
            ]
            for response in await asyncio.gather(*futures):
                results.extend(response)

    results: t.List[t.Tuple[str]] = []
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_query())

    results_df = pd.DataFrame(results)[["id", "url"]]
    return results_df.set_index("id")


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

        meta_df = pd.DataFrame(columns=["id", "url"], dtype="string")
        meta_df = meta_df.set_index("id")

        dataframe = dataframe["embeddings_data"].map_partitions(
            query_clip_client,
            client=client,
            meta=meta_df
        )

        dataframe = dataframe.rename(columns={"url": "images_url"})

        return dataframe



if __name__ == "__main__":
    component = LAIONRetrievalComponent.from_file()
    component.run()
