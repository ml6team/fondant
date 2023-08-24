"""A component that downloads common crawl files."""
import logging
import os
from typing import List, Optional

import dask.dataframe as dd
from fondant.component import DaskLoadComponent
from fsspec.implementations.http import HTTPFileSystem
from trafilatura.settings import use_config
from utils import (
    extract_bytes_from_warc_file_http,
    extract_html,
    validate_commoncrawl_index_filters,
)

logger = logging.getLogger(__name__)

CC_BASE_URL = "http://data.commoncrawl.org"


class CommonCrawlDownloadComponent(DaskLoadComponent):
    """Component that download common crawl files."""

    def __init__(self,
                 *_,
                 common_crawl_indices: List[str],
                 filters: List[dict],
                 extract_plain_text: bool,
                 n_records_to_download: Optional[int] = None):
        # init global trafilatura config for multi-processing
        self.trafilatura_config = use_config()
        self.trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
        self.filters = validate_commoncrawl_index_filters(filters)
        self.extract_plain_text = extract_plain_text
        self.n_records_to_download = n_records_to_download
        self.index_files = [self.get_http_url_path(url) for url in common_crawl_indices]

    def filter_common_crawl_index(self) -> dd.DataFrame:
        """Use common crawl index and provided filters to retrieve relevant pages
        from common crawl.
        """
        logger.info("Filtering common crawl index...")

        output_columns = [
            "url_surtkey",
            "url",
            "warc_filename",
            "warc_record_offset",
            "warc_record_length",
        ]

        https_filesytem = HTTPFileSystem()

        return dd.read_parquet(
            self.index_files,
            engine='pyarrow',
            filters=self.filters,
            columns=output_columns,
            filesystem=https_filesytem,
        )

    def download_warc_content(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """Download the content of the warc files."""
        logger.info("Downloading common crawl files...")
        return dataframe.assign(content=dataframe.apply(
            lambda row: extract_bytes_from_warc_file_http(row["warc_filename"],
                                                          row["warc_record_offset"],
                                                          row["warc_record_length"]),
            axis=1,
            meta=("content", "str")))

    def load(self) -> dd.DataFrame:
        """
        Returns:
            Dask dataframe.
        """
        ddf = self.filter_common_crawl_index()

        # Repartitioning to utilize all cores
        ddf = ddf.repartition(npartitions=os.cpu_count())

        ddf = self.download_warc_content(ddf)

        if self.extract_plain_text:
            ddf["content"] = ddf["content"].apply(
                lambda x: extract_html(x, self.trafilatura_config), meta=("content", "str"))

        ddf = ddf[['url', 'content']]

        ddf.columns = [
            "webpage_url",
            "webpage_content",
        ]

        if self.n_records_to_download is not None:
            ddf_n_partitions = ddf.partitions
            ddf = ddf.head(self.n_records_to_download)
            ddf = dd.from_pandas(ddf, npartitions=min(ddf_n_partitions, len(ddf)))

        return ddf

    @staticmethod
    def get_http_url_path(url):
        """Construct http path to common crawl index file."""
        if CC_BASE_URL in url:
            return url

        return f"{CC_BASE_URL}/{url}"
