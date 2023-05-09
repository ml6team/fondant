"""
This component downloads images based on URLs.

Some functions here are directly taken from https://github.com/rom1504/img2dataset/blob/main/img2dataset/downloader.py.
"""
import logging
import io
import traceback
import urllib

import dask.dataframe as dd

from fondant.component import FondantTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def is_disallowed(headers, user_agent_token, disallowed_header_directives):
    """Check if HTTP headers contain an X-Robots-Tag directive disallowing usage"""
    for values in headers.get_all("X-Robots-Tag", []):
        try:
            uatoken_directives = values.split(":", 1)
            directives = [x.strip().lower() for x in uatoken_directives[-1].split(",")]
            ua_token = (
                uatoken_directives[0].lower() if len(uatoken_directives) == 2 else None
            )
            if (ua_token is None or ua_token == user_agent_token) and any(
                x in disallowed_header_directives for x in directives
            ):
                return True
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"Failed to parse X-Robots-Tag: {values}: {err}")
    return False


def download_image(row, timeout, user_agent_token, disallowed_header_directives):
    """Download an image with urllib"""
    url = row.images_url
    img_stream = None
    user_agent_string = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    )
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/rom1504/img2dataset)"
    try:
        request = urllib.request.Request(
            url, data=None, headers={"User-Agent": user_agent_string}
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
    except Exception as err:  # pylint: disable=broad-except
        if img_stream is not None:
            img_stream.close()
        return None


def download_image_with_retry(
    row, timeout, retries, user_agent_token, disallowed_header_directives
):
    for _ in range(retries + 1):
        img_stream = download_image(
            row, timeout, user_agent_token, disallowed_header_directives
        )
        if img_stream is not None:
            return img_stream
    return None


class DownloadImagesComponent(FondantTransformComponent):
    """
    Component that downloads images based on URLs.
    """

    def transform(
        self, *, dataframe: dd.DataFrame, timeout: int = 10, retries: int = 0
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            timeout: Maximum time (in seconds) to wait when trying to download an image.
            retries: Number of times to retry downloading an image if it fails.

        Returns:
            Dask dataframe
        """

        dataframe["images_data"] = dataframe.apply(
            lambda example: download_image_with_retry(
                example, timeout=timeout, retries=retries
            ),
            axis=1,
            meta=("images", object),
        )

        return dataframe


if __name__ == "__main__":
    component = DownloadImagesComponent.from_file("../fondant_component.yaml")
    component.run()
