"""
This component downloads images based on URLs, and resizes them based on various settings like
minimum image size and aspect ratio.
"""
import asyncio
import io
import logging
import typing as t

import httpx
import pandas as pd
from fondant.component import PandasTransformComponent
from resizer import Resizer

logger = logging.getLogger(__name__)


class DownloadImagesComponent(PandasTransformComponent):
    """Component that downloads images based on URLs."""

    def __init__(
        self,
        *,
        timeout: int,
        retries: int,
        n_connections: int,
        image_size: int,
        resize_mode: str,
        resize_only_if_bigger: bool,
        min_image_size: int,
        max_aspect_ratio: float,
    ):
        """Component that downloads images from a list of URLs and executes filtering and resizing.

        Args:
            timeout: Maximum time (in seconds) to wait when trying to download an image.
            retries: Number of times to retry downloading an image if it fails.
            n_connections: Number of concurrent connections opened per process. Decrease this
                number if you are running into timeout errors. A lower number of connections can
                increase the success rate but lower the throughput.
            image_size: Size of the images after resizing.
            resize_mode: Resize mode to use. One of "no", "keep_ratio", "center_crop", "border".
            resize_only_if_bigger: If True, resize only if image is bigger than image_size.
            min_image_size: Minimum size of the images.
            max_aspect_ratio: Maximum aspect ratio of the images.

        Returns:
            Dask dataframe
        """
        super().__init__()
        self.timeout = timeout
        self.retries = retries
        self.n_connections = n_connections
        self.resizer = Resizer(
            image_size=image_size,
            resize_mode=resize_mode,
            resize_only_if_bigger=resize_only_if_bigger,
            min_image_size=min_image_size,
            max_aspect_ratio=max_aspect_ratio,
        )

    async def download_image(
        self,
        url: str,
        *,
        semaphore: asyncio.Semaphore,
    ) -> t.Optional[bytes]:
        url = url.strip()

        user_agent_string = (
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) "
            "Gecko/20100101 Firefox/72.0 "
            "(compatible; +https://github.com/ml6team/fondant)"
        )

        transport = httpx.AsyncHTTPTransport(retries=self.retries)
        async with httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
        ) as client:
            try:
                async with semaphore:
                    response = await client.get(
                        url,
                        timeout=self.timeout,
                        headers={"User-Agent": user_agent_string},
                    )
                image_stream = response.content
            except Exception as e:
                logger.warning(f"Skipping {url}: {repr(e)}")
                image_stream = None

        return image_stream

    async def download_and_resize_image(
        self,
        id_: str,
        url: str,
        *,
        semaphore: asyncio.Semaphore,
    ) -> t.Tuple[str, t.Optional[bytes], t.Optional[int], t.Optional[int]]:
        image_stream = await self.download_image(url, semaphore=semaphore)
        if image_stream is not None:
            image_stream, width, height = self.resizer(io.BytesIO(image_stream))
        else:
            image_stream, width, height = None, None, None
        return id_, image_stream, width, height

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Downloading {len(dataframe)} images...")

        results: t.List[t.Tuple[str, bytes, int, int]] = []

        async def download_dataframe() -> None:
            semaphore = asyncio.Semaphore(self.n_connections)

            images = await asyncio.gather(
                *[
                    self.download_and_resize_image(id_, url, semaphore=semaphore)
                    for id_, url in zip(dataframe.index, dataframe["image_url"])
                ],
            )
            results.extend(images)

        asyncio.run(download_dataframe())

        columns = ["id", "image", "image_width", "image_height"]
        if results:
            results_df = pd.DataFrame(results, columns=columns)
        else:
            results_df = pd.DataFrame(columns=columns)

        results_df = results_df.dropna()
        results_df = results_df.set_index("id", drop=True)

        return results_df
