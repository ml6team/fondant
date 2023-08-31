"""A component that downloads common crawl files."""
import asyncio
import io
import logging
import os
import typing as t

import dask
import dask.dataframe as dd
import httpx
import pandas as pd
from dask.distributed import Client
from fondant.component import DaskLoadComponent
from fsspec.implementations.http import HTTPFileSystem
from utils import (
    extract_html,
    parse_commoncrawl_index_filters,
    read_warc_content,
)

logger = logging.getLogger(__name__)

dask.config.set({"distributed.worker.daemon": False})
Client()

CC_BASE_URL = "http://data.commoncrawl.org"


class CommonCrawlDownloadComponent(DaskLoadComponent):
    """Component that download common crawl files."""

    def __init__(
        self,
        *_,
        common_crawl_indices: t.List[str],
        filters: t.List[dict],
        extract_plain_text: bool,
        n_records_to_download: t.Optional[int] = None,
    ):
        self.filters = parse_commoncrawl_index_filters(filters) if filters else None
        self.extract_plain_text = extract_plain_text
        self.n_records_to_download = n_records_to_download
        self.index_files = [self.get_http_url_path(url) for url in common_crawl_indices]

    def load_index(self) -> dd.DataFrame:
        """
        Load the common crawl index with the provided filters.

        Returns:
            A dataframe containing the filtered urls and their location in the WARC archivei
        """
        logger.info("Loading filtered common crawl index...")

        output_columns = [
            "url_surtkey",
            "url",
            "warc_filename",
            "warc_record_offset",
            "warc_record_length",
        ]

        https_filesytem = HTTPFileSystem()

        dataframe = dd.read_parquet(
            self.index_files,
            filters=self.filters,
            columns=output_columns,
            filesystem=https_filesytem,
        )

        return dataframe.set_index("url_surtkey", sorted=True, drop=True)

    async def download_warc_content(
        self,
        row: t.Any,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> t.Tuple[str, str, t.Optional[str]]:
        """
        Download content of a single web page.

        Args:
            row: This should be a NamedTuple returned by df.itertuples(), but cannot be
                 typehinted as such.
            client: Httpx client to use
            semaphore: Semaphore to limit amount of concurrent requests

        Returns:
            A tuple containing the index, url, and extracted content
        """
        url = f"{CC_BASE_URL}/{row.warc_filename}"
        headers = {
            "Range": f"bytes={row.warc_record_offset}-"
                     f"{row.warc_record_offset + row.warc_record_length - 1}",
        }

        try:
            async with semaphore:
                response = await client.get(url, headers=headers)
        except Exception as e:
            logger.warning(f"Error downloading {url} with headers {headers}: {repr(e)}")
            return row.Index, url, None
        else:
            with io.BytesIO(response.content) as warc_stream:
                content = read_warc_content(warc_stream)

            if self.extract_plain_text and content is not None:
                content = extract_html(content)

            return row.Index, row.url, content

    def download_and_extract(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Concurrently download and extract the WARC files referenced in the provided dataframe."""
        html_content = []

        async def download_dataframe() -> None:
            semaphore = asyncio.Semaphore(20)

            transport = httpx.AsyncHTTPTransport(retries=1)
            async with httpx.AsyncClient(transport=transport, timeout=10) as client:
                html = await asyncio.gather(
                    *[self.download_warc_content(row, client=client, semaphore=semaphore)
                      for row in dataframe.itertuples()],
                )
                html_content.extend(html)

        asyncio.run(download_dataframe())

        columns = ["url_surtkey", "url", "content"]
        if html_content:
            content_df = pd.DataFrame(html_content, columns=columns)
        else:
            content_df = pd.DataFrame(columns=columns)

        content_df = content_df.dropna()
        content_df = content_df.set_index("url_surtkey", drop=True)

        return content_df

    def load(self) -> dd.DataFrame:
        index_ddf = self.load_index()

        # Repartition to parallelize and reduce memory footprint of each partition.
        # Each (unfiltered) index file contains around ~10M urls. The numbers below are chosen so
        # they end up with around 10k urls per partition. With an estimated content of ~25KB per
        # page, this brings us close to the recommended partition size of 250MB.
        if self.n_records_to_download is not None:
            n_partitions = max(os.cpu_count(), self.n_records_to_download // 10000)  # type: ignore
            logging.info(f"Repartitioning to {n_partitions} partitions.")
            index_ddf = dd.from_pandas(index_ddf.head(self.n_records_to_download,
                                                      npartitions=-1), npartitions=n_partitions)
        else:
            n_partitions = len(self.index_files) * 1000
            logging.info(f"Repartitioning to {n_partitions} partitions.")
            index_ddf = index_ddf.repartition(npartitions=n_partitions)

        meta = pd.DataFrame(columns=["url", "content"])

        content_ddf = index_ddf.map_partitions(
            self.download_and_extract,
            meta=meta,
        )

        content_ddf.columns = [
            "webpage_url",
            "webpage_content",
        ]

        return content_ddf

    @staticmethod
    def get_http_url_path(url):
        """Construct http path to common crawl index file."""
        if CC_BASE_URL in url:
            return url

        return f"{CC_BASE_URL}/{url}"
