"""This component retrieves image URLs from LAION-5B based on a set of seed prompts."""
import asyncio
import concurrent.futures
import logging
import typing as t

import pandas as pd
from clip_client import ClipClient, Modality

from fondant.component import PandasTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class LAIONRetrievalComponent(PandasTransformComponent):
    """Component that retrieves image URLs from LAION-5B based on a set of prompts."""

    def setup(
            self,
            *,
            num_images: int,
            aesthetic_score: int,
            aesthetic_weight: float
    ) -> None:
        """

        Args:
            num_images: number of images to retrieve for each prompt
            aesthetic_score: ranking score for aesthetic embedding, higher is prettier,
                between 0 and 9.
            aesthetic_weight: weight of the aesthetic embedding to add to the query,
                between 0 and 1.
        """
        self.client = ClipClient(
            url="https://knn.laion.ai/knn-service",
            indice_name="laion5B-L-14",
            num_images=num_images,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
            modality=Modality.IMAGE,
        )

    def transform(
            self,
            dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Args:
            dataframe: Pandas dataframe.

        Returns:
            Dask dataframe
        """
        results: t.List[t.Tuple[str]] = []
        loop = asyncio.new_event_loop()

        async def async_query():
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    loop.run_in_executor(
                        executor,
                        self.client.query,
                        prompt
                    )
                    for prompt in dataframe["prompts_text"]
                ]
                for response in await asyncio.gather(*futures):
                    results.extend(response)

        loop.run_until_complete(async_query())

        results_df = pd.DataFrame(results)[["id", "url"]]
        results_df.rename(columns={"url": "images_url"}, inplace=True)
        results_df.set_index("id", inplace=True)

        return results_df


if __name__ == "__main__":
    component = LAIONRetrievalComponent.from_args()
    component.run()
