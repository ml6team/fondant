"""This component retrieves image URLs from LAION-5B based on a set of seed prompts."""
import asyncio
import concurrent.futures
import logging
import typing as t

import pandas as pd
from clip_client import ClipClient, Modality
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class LAIONRetrievalComponent(PandasTransformComponent):
    """Component that retrieves image URLs from LAION-5B based on a set of prompts."""

    def __init__(
        self,
        *_,
        num_images: int,
        aesthetic_score: int,
        aesthetic_weight: float,
        url: str,
    ) -> None:
        """

        Args:
            num_images: number of images to retrieve for each prompt
            aesthetic_score: ranking score for aesthetic embedding, higher is prettier,
                between 0 and 9.
            aesthetic_weight: weight of the aesthetic embedding to add to the query,
                between 0 and 1.
            url: The url of the backend clip retrieval service, defaults to the public clip url.
        """
        self.client = ClipClient(
            url=url,
            indice_name="laion5B-L-14",
            num_images=num_images,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
            modality=Modality.IMAGE,
        )

    def query(self, id_: t.Any, prompt: str) -> t.List[t.Dict]:
        results = self.client.query(text=prompt)
        return [dict(d, prompt_id=id_) for d in results]

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        results: t.List[t.Tuple[str]] = []
        loop = asyncio.new_event_loop()

        async def async_query():
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    loop.run_in_executor(
                        executor,
                        self.query,
                        row.id,
                        row.prompts_text,
                    )
                    for row in dataframe.itertuples()
                ]
                for response in await asyncio.gather(*futures):
                    results.extend(response)

        loop.run_until_complete(async_query())

        results_df = pd.DataFrame(results)[["id", "url", "prompt_id"]]
        results_df = results_df.set_index("id")

        results_df.rename(columns={"url": "images_url"})

        return results_df
