import logging
from typing import List

import io
import boto3

import dask.bag as db
import dask.dataframe as dd
from dask.delayed import delayed

import pandas as pd

import requests
from warcio.archiveiterator import ArchiveIterator

from fondant.component import DaskTransformComponent

logger = logging.getLogger(__name__)

BASE_URL = "https://data.commoncrawl.org/"


def get_records_from_warc_file(warc_file: str, n_records_to_download: int) -> List:
    """Downloads a WARC file and extracts the webpages.
    Args:
        warc_file: The path to the WARC file.
        n_records_to_download: The number of webpages to download from the WARC file.
    Returns:
        A list of webpages.
    """
    logger.info(f"Processing WARC file from segment path: {warc_file}...")
    records = []
    counter = 0
    response = requests.get(BASE_URL + warc_file, stream=True)
    response.raise_for_status()

    for record in ArchiveIterator(response.raw, arc2warc=True):
        if record.rec_type == "response":
            url = record.rec_headers.get_header("WARC-Target-URI")
            content = (
                record.content_stream()
                .read()
                .decode(errors="replace", encoding="utf-8")
            )
            records.append([url, content])
            counter += 1

        if counter == n_records_to_download:
            break

    return records


class DownloadCommoncrawlSegments(DaskTransformComponent):
    def transform(self, df: dd.DataFrame, n_records_to_download: int) -> dd.DataFrame:
        """Downloads Commoncrawl segments based on a list of WARC paths.
        Args:
            df: A Dask DataFrame containing a column of WARC paths.
            n_webpages_to_download: The number of webpages to download from each segment.
        Returns:
            A Dask DataFrame containing the downloaded webpages.
        """
        segment_paths = df["segment_path"].to_bag()

        records = segment_paths.map(get_records_from_warc_file, n_records_to_download)

        flattened_records = records.flatten()
        dask_df = flattened_records.to_dataframe(
            columns=["webpage_url", "webpage_html"]
        )

        return dask_df


if __name__ == "__main__":
    component = DownloadCommoncrawlSegments.from_args()
    component.run()
