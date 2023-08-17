"""
This component downloads images based on URLs, and resizes them based on various settings like
minimum image size and aspect ratio.

Some functions here are directly taken from
https://github.com/rom1504/img2dataset/blob/main/img2dataset/downloader.py.
"""
import asyncio
import concurrent.futures
import functools
import io
import logging
import traceback
import typing as t
import urllib

import pandas as pd
from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor
from resizer import Resizer

logger = logging.getLogger(__name__)


def is_disallowed(headers, user_agent_token, disallowed_header_directives):
    """Check if HTTP headers contain an X-Robots-Tag directive disallowing usage."""
    for values in headers.get_all("X-Robots-Tag", []):
        try:
            uatoken_directives = values.split(":", 1)
            directives = [x.strip().lower() for x in uatoken_directives[-1].split(",")]
            ua_token = (
                uatoken_directives[0].lower() if len(uatoken_directives) == 2  # noqa: PLR2004
                else None
            )
            if (ua_token is None or ua_token == user_agent_token) and any(
                    x in disallowed_header_directives for x in directives
            ):
                return True
        except Exception as err:
            traceback.print_exc()
            print(f"Failed to parse X-Robots-Tag: {values}: {err}")
    return False


def download_image(url, timeout, user_agent_token, disallowed_header_directives):
    """Download an image with urllib."""
    img_stream = None
    user_agent_string = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    )
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; " \
                             f"+https://github.com/ml6team/fondant)"
    try:
        request = urllib.request.Request(
            url, data=None, headers={"User-Agent": user_agent_string},
        )
        with urllib.request.urlopen(request, timeout=timeout) as r:
            if disallowed_header_directives and is_disallowed(
                    r.headers,
                    user_agent_token,
                    disallowed_header_directives,
            ):
                return None
            img_stream = io.BytesIO(r.read())
        return img_stream
    except Exception:
        if img_stream is not None:
            img_stream.close()
        return None


def download_image_with_retry(
        id_,
        url,
        *,
        timeout,
        retries,
        resizer,
        user_agent_token=None,
        disallowed_header_directives=None,
):
    for _ in range(retries + 1):
        img_stream = download_image(
            url, timeout, user_agent_token, disallowed_header_directives,
        )
        if img_stream is not None:
            # resize the image
            img_str, width, height = resizer(img_stream)
            return id_, img_str, width, height
    return id_, None, None, None


class DownloadImagesComponent(PandasTransformComponent):
    """Component that downloads images based on URLs."""

    def __init__(self,
                 *_,
                 timeout: int,
                 retries: int,
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
            image_size: Size of the images after resizing.
            resize_mode: Resize mode to use. One of "no", "keep_ratio", "center_crop", "border".
            resize_only_if_bigger: If True, resize only if image is bigger than image_size.
            min_image_size: Minimum size of the images.
            max_aspect_ratio: Maximum aspect ratio of the images.

        Returns:
            Dask dataframe
        """
        self.timeout = timeout
        self.retries = retries
        self.resizer = Resizer(
            image_size=image_size,
            resize_mode=resize_mode,
            resize_only_if_bigger=resize_only_if_bigger,
            min_image_size=min_image_size,
            max_aspect_ratio=max_aspect_ratio,
        )

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        logger.info(f"Downloading {len(dataframe)} images...")

        results: t.List[t.Tuple[str]] = []
        loop = asyncio.new_event_loop()

        async def async_download():
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    loop.run_in_executor(
                        executor,
                        functools.partial(
                            download_image_with_retry,
                            id_,
                            url,
                            timeout=self.timeout,
                            retries=self.retries,
                            resizer=self.resizer,
                        ),
                    )
                    for id_, url in zip(dataframe.index, dataframe["images"]["url"])
                ]
                for response in await asyncio.gather(*futures):
                    results.append(response)

        loop.run_until_complete(async_download())

        results_df = pd.DataFrame(results)
        results_df.columns = ["id", ("images", "data"), ("images", "width"), ("images", "height")]
        results_df.set_index("id", drop=True)

        return results_df


if __name__ == "__main__":
    executor = PandasTransformExecutor.from_args()
    executor.execute(DownloadImagesComponent)
