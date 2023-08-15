"""This component loads a dataset from CommonCrawl based on a given index."""
import logging
import typing as t

import io
import boto3
import gzip

import dask.dataframe as dd
import pandas as pd

from fondant.component import DaskLoadComponent
from fondant.executor import DaskLoadExecutor

logger = logging.getLogger(__name__)

S3_COMMONCRAWL_BUCKET = "commoncrawl"


def fetch_warc_file_from_s3(
    s3_bucket: str,
    s3_key: str,
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
) -> bytes:
    """Fetches a WARC file from S3 and returns its content as a Dask DataFrame.
    Args:
        s3_bucket: The name of the S3 bucket.
        s3_key: The key of the S3 object to be downloaded.
        aws_access_key_id: AWS access key id for initialise boto client
        aws_secret_access_key: AWS access key id for initialise boto client

    Returns:
        File object containing the WARC file content.
    """
    logger.info(f"Fetching WARC file from S3: {s3_bucket}/{s3_key}...")

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
    )
    s3 = session.client("s3")

    file_obj = io.BytesIO()
    s3.download_fileobj(s3_bucket, s3_key, file_obj)
    file_obj.seek(0)

    return file_obj


def read_warc_paths_file(
    warc_file: bytes,
    n_segments_to_load: t.Optional[int] = None,
    offset: t.Optional[int] = 0,
) -> dd.DataFrame:
    """Reads a WARC file containing a list of segment file paths and returns a Dask DataFrame.
    Args:
        warc_file: The WARC file to read.
        n_segments_to_load: The number of segments to load from the WARC file.
        offset: The offset from which to start loading segments.
    Returns:
        A Dask DataFrame containing the segment file paths.
    """
    logger.info(f"Reading WARC file...")
    warc_paths = []
    with gzip.open(warc_file, mode="rt") as f:
        warc_paths = [line.strip() for line in f]

    df = pd.DataFrame(warc_paths, columns=["warc_paths"])

    if n_segments_to_load:
        logger.info(f"Loading {n_segments_to_load} segments from offset {offset}...")
        df = df.iloc[offset : offset + n_segments_to_load]

    dask_df = dd.from_pandas(df, npartitions=1)
    dask_df = dask_df.rename(columns={"warc_paths": "segment_path"})

    return dask_df


class LoadFromCommonCrawlComponent(DaskLoadComponent):
    def __init__(
        self,
        *args,
        index_name: str,
        n_segments_to_load: t.Optional[int] = None,
        offset: t.Optional[int] = 0,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
    ) -> None:
        self.index_name = index_name
        self.n_segments_to_load = n_segments_to_load
        self.offset = offset
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        """Loads a dataset of segment file paths from CommonCrawl based on a given index.
        
        Args:
            index_name: The name of the CommonCrawl index to load.
            n_segments_to_load: The number of segments to load from the index.
            offset: The offset from which to start loading segments.
            aws_access_key_id: AWS access key id for initialise boto client
            aws_secret_access_key: AWS access key id for initialise boto client
        """

    def load(self) -> dd.DataFrame:
        """
        Returns:
            A Dask DataFrame containing the segment file paths.
        """
        logger.info(f"Loading CommonCrawl index {self.index_name}...")
        warc_paths_file_key = f"crawl-data/{self.index_name}/warc.paths.gz"
        warc_paths_file_content = fetch_warc_file_from_s3(
            S3_COMMONCRAWL_BUCKET,
            warc_paths_file_key,
            self.aws_access_key_id,
            self.aws_secret_access_key,
        )

        warc_paths_df = read_warc_paths_file(
            warc_paths_file_content,
            self.n_segments_to_load,
            self.offset,
        )

        return warc_paths_df


if __name__ == "__main__":
    executor = DaskLoadExecutor.from_args()
    executor.execute(LoadFromCommonCrawlComponent)
