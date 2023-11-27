import tempfile
import uuid

import dask.dataframe as dd

from src.main import IndexQdrantComponent, QdrantClient, models


def test_qdrant_write():
    """
    Test case for the write method of the IndexQdrantComponent class.

    This test creates a temporary collection using a QdrantClient.
    Writes data to it using the write method of the IndexQdrantComponent.
    Asserts that the count of entries in the collection is equal to the expected number of entries.
    """
    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=str(tmpdir))
        entries = 100

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(distance=models.Distance.COSINE, size=5),
        )
        # There cannot be multiple clients accessing the same local persistent storage
        # Qdrant server supports multiple concurrent access
        del client

        component = IndexQdrantComponent(
            collection_name=collection_name,
            path=str(tmpdir),
        )

        dask_dataframe = dd.DataFrame.from_dict(
            {
                "text": [
                    "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo",
                ]
                * entries,
                "embedding": [[0.1, 0.2, 0.3, 0.4, 0.5]] * entries,
            },
            npartitions=1,
        )

        component.write(dask_dataframe)
        del component

        client = QdrantClient(path=str(tmpdir))
        assert client.count(collection_name).count == entries
