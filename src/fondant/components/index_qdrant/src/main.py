import ast
from typing import List, Optional

import dask.dataframe as dd
from fondant.component import DaskWriteComponent
from qdrant_client import QdrantClient, models
from qdrant_client.qdrant_fastembed import uuid


class IndexQdrantComponent(DaskWriteComponent):
    def __init__(
        self,
        *,
        collection_name: str,
        location: Optional[str] = None,
        batch_size: int = 64,
        parallelism: int = 1,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
    ):
        """Initialize the IndexQdrantComponent with the component parameters."""
        super().__init__()
        self.client = QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            force_disable_check_same_thread=force_disable_check_same_thread,
        )
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.parallelism = parallelism

    def teardown(self) -> None:
        self.client.close()

    def write(self, dataframe: dd.DataFrame) -> None:
        """
        Writes the data from the given Dask DataFrame to the Qdrant collection.

        Args:
            dataframe (dd.DataFrame): The Dask DataFrame containing the data to be written.
        """
        records: List[models.Record] = []
        for part in dataframe.partitions:
            df = part.compute()
            for row in df.itertuples():
                payload = {
                    "id_": str(row.Index),
                    "passage": row.text,
                }
                id = str(uuid.uuid4())
                # Check if 'text_embedding' attribute is a string.
                # If it is, safely evaluate and convert it into a list of floats.
                # else (i.e., it is already a list), it is directly assigned.
                embedding = (
                    ast.literal_eval(row.embedding)
                    if isinstance(row.embedding, str)
                    else row.embedding
                )
                records.append(models.Record(id=id, payload=payload, vector=embedding))

            self.client.upload_records(
                collection_name=self.collection_name,
                records=records,
                batch_size=self.batch_size,
                parallel=self.parallelism,
                wait=True,
            )
