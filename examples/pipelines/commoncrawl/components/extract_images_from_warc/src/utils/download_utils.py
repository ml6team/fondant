import logging

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


logger = logging.getLogger(__name__)

COMMONCRAWL_BASE_URL = "https://data.commoncrawl.org/"


def download_warc_file(
    warc_file: str, retries: int = 10, backoff_factor: int = 5
) -> requests.Response:
    """Downloads a WARC file using http requests.
    Args:
        warc_file: The path to the WARC file.
    Returns:
        The WARC file as a bytes object.
    """
    session = Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[502, 503, 504],
        allowed_methods={"POST", "GET"},
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    try:
        response = session.get(COMMONCRAWL_BASE_URL + warc_file, stream=True)
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"Error downloading WARC file: {e}")
        raise
