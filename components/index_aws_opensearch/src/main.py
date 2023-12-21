from typing import Dict, Any
import dask.dataframe as dd
from fondant.component import DaskWriteComponent
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

class IndexAWSOpenSearchComponent(DaskWriteComponent):
    def __init__(
        self,
        host: str,
        region: str,
        index_name: str,
        port: int = 443,
        use_ssl: bool = True,
        verify_certs: bool = True,
        pool_maxsize: int = 20,
        **kwargs,
    ):
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
            **kwargs,
        )

    def create_index(self, index_body: Dict[str, Any]):
        """Creates an index in AWS OpenSearch

        Args:
            index_body (Dict[str, Any]): Parameters that specify index settings, mappings, and aliases for newly created index.
        """
        response = self.client.indices.create(self.index_name, body=index_body)

    def write(self, dataframe: dd.DataFrame):
        """
        Writes the data from the given Dask DataFrame to AWS OpenSearch Index.

        Args:
            index_name (str): The existing index to which the given dataframe will be written
            dataframe (dd.DataFrame): The Dask DataFrame containing the data to be written.
        """
        if not self.client.indices.exists(index=self.index_name):
            raise ValueError(f"Index: {self.index_name} doesn't exist. Please Create")

        for part in dataframe.partitions:
            df = part.compute()
            for row in df.itertuples():
                body = {
                    "embedding": row.embedding,
                    "text": row.text
                }
                response = self.client.index(
                    index=self.index_name, id=str(row.Index), body=body
                )