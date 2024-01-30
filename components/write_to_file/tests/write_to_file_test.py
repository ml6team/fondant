import tempfile
import uuid

import dask.dataframe as dd

from src.main import IndexQdrantComponent, QdrantClient, models


def test_write_to_csv():
    """
    Test case for write to file component
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        entries = 10


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
