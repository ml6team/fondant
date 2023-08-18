import time
import logging

import boto3

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from requests import RequestException, ConnectionError
from urllib3.util import Retry


logger = logging.getLogger(__name__)

S3_COMMONCRAWL_BUCKET = "commoncrawl"
COMMONCRAWL_BASE_URL = "https://data.commoncrawl.org/"


def get_warc_file_using_boto3(s3_key: str, s3_client) -> bytes:
    """Downloads a WARC file using boto3.
    Args:
        warc_file: The path to the WARC file.
    Returns:
        The WARC file as a bytes object.
    """
    response = s3_client.get_object(Bucket=S3_COMMONCRAWL_BUCKET, Key=s3_key)
    return response["Body"]


def get_warc_file_using_requests(
    warc_file: str, retries: int = 3, backoff_factor: int = 5
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading WARC file: {e}")
        raise
