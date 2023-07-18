import time
import logging

import boto3

import requests
from requests import RequestException, ConnectionError

logger = logging.getLogger(__name__)

S3_COMMONCRAWL_BUCKET = "commoncrawl"
COMMONCRAWL_BASE_URL = "https://data.commoncrawl.org/"


def get_warc_file_using_boto3(s3_key: str) -> bytes:
    """Downloads a WARC file using boto3.
    Args:
        warc_file: The path to the WARC file.
    Returns:
        The WARC file as a bytes object.
    """
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=S3_COMMONCRAWL_BUCKET, Key=s3_key)
    return response["Body"]


def get_warc_file_using_requests(warc_file: str) -> requests.Response:
    retry = 0
    retries = 3
    retry_delay = 5

    while retry < retries:
        try:
            response = requests.get(COMMONCRAWL_BASE_URL + warc_file, stream=True)
            response.raise_for_status()
            return response
        except (RequestException, ConnectionError) as e:
            logger.error(f"Error downloading WARC file: {e}")
            logger.error(f"Retrying... {retry}/{retries}")
            time.sleep(retry_delay)
            retry += 1
    raise Exception(f"Failed to download WARC file after multiple retries: {warc_file}")
