"""This component loads a dataset from CommonCrawl based on a given index."""
import logging
import typing as t

import io
import boto3
import gzip

import dask.dataframe as dd
import pandas as pd

from fondant.component import LoadComponent

logger = logging.getLogger(__name__)

S3_BASE_URL = "s3://commoncrawl/crawl-data"
S3_COMMONCRAWL_BUCKET = "commoncrawl"


def fetch_warc_file_from_s3(s3_bucket: str, s3_key) -> dd.DataFrame:
    """Fetches a WARC file from S3 and returns its content as a Dask DataFrame."""
    logger.info(f"Fetching WARC file from S3: {s3_bucket}/{s3_key}...")

    s3 = boto3.client("s3")
    file_obj = io.BytesIO()
    s3.download_fileobj(s3_bucket, s3_key, file_obj)
    file_obj.seek(0)

    return file_obj


def read_warc_paths_file(
    warc_file: bytes, n_segments_to_load: t.Optional[int] = None
) -> dd.DataFrame:
    """Reads a WARC file and returns its content as a Dask DataFrame."""
    logger.info(f"Reading WARC file...")
    warc_paths = []
    with gzip.open(warc_file, mode="rt") as f:
        warc_paths = [line.strip() for line in f]

    df = pd.DataFrame(warc_paths, columns=["warc_paths"])
    dask_df = dd.from_pandas(df, npartitions=1)
    dask_df = dask_df.rename(columns={"warc_paths": "segment_path"})

    if n_segments_to_load:
        dask_df = dask_df.head(n_segments_to_load)
        dask_df = dd.from_pandas(dask_df, npartitions=1)

    return dask_df


class LoadFromCommonCrawl(LoadComponent):
    def load(
        self, index_name: str, n_segments_to_load: t.Optional[int] = None
    ) -> dd.DataFrame:
        logger.info(f"Loading CommonCrawl index {index_name}...")
        warc_paths_file_key = f"crawl-data/{index_name}/warc.paths.gz"
        warc_paths_file_content = fetch_warc_file_from_s3(
            S3_COMMONCRAWL_BUCKET, warc_paths_file_key
        )

        warc_paths_df = read_warc_paths_file(
            warc_paths_file_content, n_segments_to_load
        )

        return warc_paths_df


if __name__ == "__main__":
    component = LoadFromCommonCrawl.from_args()
    component.run()
