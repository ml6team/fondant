import os
import logging
from typing import List, Optional

import dask.dataframe as dd
import pandas as pd

import gzip
from warcio.archiveiterator import ArchiveIterator

from fondant.component import DaskTransformComponent
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
        response = get_warc_file_using_requests(warc_file)
        return get_records(response.raw, get_plain_text, n_records_to_download)


class DownloadCommoncrawlSegments(DaskTransformComponent):
    def transform(
        self,
        df: dd.DataFrame,
        use_s3: Optional[bool] = False,
        get_plain_text: Optional[bool] = False,
        n_records_to_download: Optional[int] = None,
        partition_size: Optional[int] = None,
    ) -> dd.DataFrame:
        """Downloads Commoncrawl segments based on a list of WARC paths.
        Args:
            df: A Dask DataFrame containing a column of WARC paths.
            use_s3: Whether to download the WARC files from S3 or from the Commoncrawl API.
            get_plain_text: Whether to convert the HTML content to plain text.
            n_records_to_download: The number of webpages to download from each segment.
        Returns:
            A Dask DataFrame containing the downloaded webpages.
        """
        n_partitions = df.npartitions
        n_workers = os.cpu_count()

        if n_partitions < n_workers:
            df = df.repartition(npartitions=n_workers)

        df = (
            df.apply(
                lambda row: get_records_from_warc_file(
                    row["segment_path"], use_s3, get_plain_text, n_records_to_download
                ),
                axis=1,
                meta=("object"),
            )
            .explode()
            .apply(pd.Series, meta={0: "object", 1: "object"})
        )

        df.columns = [
            "webpage_url",
            "webpage_html",
        ]

        if partition_size:
            df = df.repartition(partition_size=f"{partition_size}MB")

        df = df.reset_index(drop=True)

        return df


if __name__ == "__main__":
    component = DownloadCommoncrawlSegments.from_args()
    component.run()
