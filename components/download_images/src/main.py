"""
This component downloads images based on URLs, and resizes them based on various settings like
minimum image size and aspect ratio.

Some functions here are directly taken from
https://github.com/rom1504/img2dataset/blob/main/img2dataset/downloader.py.
"""
import io
import logging
import traceback
import urllib

import pandas as pd
from resizer import Resizer

from fondant.component import PandasTransformComponent

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
                             f"+https://github.com/rom1504/img2dataset)"
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
            return resizer(img_stream)
    return None, None, None


class DownloadImagesComponent(PandasTransformComponent):
    """Component that downloads images based on URLs."""

    def setup(
        self,
        *,
        timeout: int = 10,
        retries: int = 0,
        image_size: int = 256,
        resize_mode: str = "border",
        resize_only_if_bigger: bool = False,
        min_image_size: int = 0,
        max_aspect_ratio: float = float("inf"),
    ):
        logger.info("Instantiating resizer...")
        self.resizer = Resizer(
            image_size=image_size,
            resize_mode=resize_mode,
            resize_only_if_bigger=resize_only_if_bigger,
            min_image_size=min_image_size,
            max_aspect_ratio=max_aspect_ratio,
        )
        self.timeout = timeout
        self.retries = retries

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[[
            ("images", "data"),
            ("images", "width"),
            ("images", "height"),
        ]] = dataframe.apply(
            lambda example: download_image_with_retry(
                url=example["images"]["url"],
                timeout=self.timeout,
                retries=self.retries,
                resizer=self.resizer,
            ),
            axis=1,
            result_type="expand",
        )

        return dataframe


if __name__ == "__main__":
    component = DownloadImagesComponent.from_args()
    component.run()
