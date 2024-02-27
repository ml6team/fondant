import logging
from typing import Any, Dict, Optional

import boto3
import dask.dataframe as dd
from fondant.component import DaskWriteComponent
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IndexAWSOpenSearchComponent(DaskWriteComponent):
    def __init__(
        self,
        *,
        host: str,
        region: str,
        index_name: str,
        index_body: Dict[str, Any],
        port: Optional[int],
        use_ssl: Optional[bool],
        verify_certs: Optional[bool],
        pool_maxsize: Optional[int],
    ):
        super().__init__()
        session = boto3.Session()
        credentials = session.get_credentials()
        auth = AWSV4SignerAuth(credentials, region)
        self.index_name = index_name
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            connection_class=RequestsHttpConnection,
            pool_maxsize=pool_maxsize,
        )
        self.create_index(index_body)

    def teardown(self, _) -> None:
        self.client.close()

    def create_index(self, index_body: Dict[str, Any]):
        """Creates an index if not existing in AWS OpenSearch.

        Args:
            index_body: Parameters that specify index settings,
            mappings, and aliases for newly created index.
        """
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index: {self.index_name} already exists.")
        else:
            logger.info(f"Creating Index: {self.index_name} with body: {index_body}")
            self.client.indices.create(index=self.index_name, body=index_body)

    def write(self, dataframe: dd.DataFrame):
        """
        Writes the data from the given Dask DataFrame to AWS OpenSearch Index.

        Args:
            dataframe: The Dask DataFrame containing the data to be written.
        """
        for part in tqdm(dataframe.partitions):
            df = part.compute()
            for row in df.itertuples():
                body = {"embedding": row.embedding, "text": row.text}
                self.client.index(
                    index=self.index_name,
                    id=str(row.Index),
                    body=body,
                )
