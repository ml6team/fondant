import os
import logging
from typing import List, Optional

import dask.dataframe as dd
import dask.delayed as delayed
import pandas as pd

import gzip
from warcio.archiveiterator import ArchiveIterator

from fondant.component import DaskTransformComponent
from fondant.executor import DaskTransformExecutor

from utils.text_utils import convert_to_plain_text
from utils.download_utils import get_warc_file_using_boto3, get_warc_file_using_requests

logger = logging.getLogger(__name__)


def get_records(file, get_plain_text, n_records_to_download) -> List[List[str]]:
    """Extracts records from a WARC file, optionally converting HTML to plain text.
    Args:
        file: The WARC file.
        get_plain_text: Whether to convert HTML to plain text.
        n_records_to_download: The number of records to download.
    Returns:
        A list of webpage records, where each record is a url and content.
    """
    records = []
    counter = 0

    for record in ArchiveIterator(file, arc2warc=True):
        if record.rec_type == "response":
            url = record.rec_headers.get_header("WARC-Target-URI")
            content = record.content_stream().read().decode("utf-8", "replace")
            if get_plain_text:
                content = convert_to_plain_text(content)
            records.append([url, content])
            counter += 1

        if n_records_to_download and counter >= n_records_to_download:
            break

    return records


def get_records_from_warc_file(
    warc_file: str,
    use_s3: Optional[bool] = False,
    get_plain_text: Optional[bool] = False,
    n_records_to_download: Optional[int] = None,
    retries: Optional[int] = None,
    backoff_factor: Optional[int] = None,
) -> List[List[str]]:
    """Downloads a WARC file and extracts the webpages.
    Args:
        warc_file: The path to the WARC file.
        use_s3: Whether to download the WARC file from S3 or from the Commoncrawl API.
        get_plain_text: Whether to convert the HTML content to plain text.
        n_records_to_download: The number of webpages to download from the WARC file.
    Returns:
        A list of webpages.
    """
    logger.info(f"Processing WARC file from segment path: {warc_file}...")

    if use_s3:
        response = get_warc_file_using_boto3(warc_file)
        with gzip.GzipFile(fileobj=response, mode="rb") as file:
            return get_records(file, get_plain_text, n_records_to_download)
    else:
        response = get_warc_file_using_requests(warc_file, retries, backoff_factor)
        return get_records(response.raw, get_plain_text, n_records_to_download)


class DownloadCommoncrawlSegments(DaskTransformComponent):
    def __init__(
        self,
        *_,
        use_s3: Optional[bool] = False,
        get_plain_text: Optional[bool] = False,
        n_records_to_download: Optional[int] = None,
        retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
    ):
        """Downloads Commoncrawl segments based on a list of WARC paths.
        Args:
            use_s3: Whether to download the WARC files from S3 or from the Commoncrawl API.
            get_plain_text: Whether to convert the HTML content to plain text.
            n_records_to_download: The number of webpages to download from each segment.
        """
        self.use_s3 = use_s3
        self.get_plain_text = get_plain_text
        self.n_records_to_download = n_records_to_download
        self.retries = retries
        self.backoff_factor = backoff_factor

    def transform(
        self,
        dataframe: dd.DataFrame,
    ) -> dd.DataFrame:
        """Downloads Commoncrawl segments based on a list of WARC paths.
        Args:
            dataframe: A Dask DataFrame containing a column of WARC paths.
        Returns:
            A Dask DataFrame containing the downloaded webpages.
        """
        segment_paths = dataframe["segment_path"].to_bag()

        records = segment_paths.map(
            get_records_from_warc_file,
            use_s3=self.use_s3,
            get_plain_text=self.get_plain_text,
            n_records_to_download=self.n_records_to_download,
            retries=self.retries,
            backoff_factor=self.backoff_factor,
        )

        flattened_records = records.flatten()
        meta = {"url": "object", "content": "object"}

        dask_df = flattened_records.to_dataframe(
            meta=meta,
        )
        dask_df = dask_df.rename(
            columns={"url": "webpage_url", "content": "webpage_html"}
        )

        logger.info(f"Downloaded {len(dask_df)} webpages from Commoncrawl.")

        return dask_df


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(DownloadCommoncrawlSegments)
