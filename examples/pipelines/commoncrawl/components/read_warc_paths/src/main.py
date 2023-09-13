"""A component that downloads common crawl files."""
import gzip
import logging
import typing as t

import dask.dataframe as dd
import pandas as pd
import requests
from fondant.component import DaskLoadComponent

logger = logging.getLogger(__name__)


class ReadWarcPathsComponent(DaskLoadComponent):
    """Component that download common crawl files."""

    def __init__(
        self,
        *_,
        common_crawl_indices: t.List[str],
        n_records_to_download: t.Optional[int] = None,
    ):
        self.index_urls = [
            self.build_index_url(index_name) for index_name in common_crawl_indices
        ]
        self.n_records_to_download = n_records_to_download

    @staticmethod
    def build_index_url(index_name: str) -> str:
        return f"http://data.commoncrawl.org/crawl-data/{index_name}/warc.paths.gz"

    def load(self) -> dd.DataFrame:
        warc_urls = []

        for index_url in self.index_urls:
            response = requests.get(index_url)
            extracted = gzip.decompress(response.content)
            warc_urls.extend([line.decode() for line in extracted.split(b"\n")])

        df = pd.DataFrame(warc_urls, columns=["warc_url"])
        if self.n_records_to_download is not None:
            df = df.head(self.n_records_to_download)

        return dd.from_pandas(df, npartitions=len(df) // 100)
