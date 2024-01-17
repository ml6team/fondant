import typing as t

import dask.dataframe as dd
import numpy as np
import pandas as pd
import weaviate
from weaviate.embedded import EmbeddedOptions

from src.main import IndexWeaviateComponent


def get_written_objects(
    client: weaviate.Client,
    class_name: str,
) -> t.List[t.Dict[str, t.Any]]:
    """Taken from https://weaviate.io/developers/weaviate/manage-data/read-all-objects."""
    query = (
        client.query.get(
            class_name,
            ["id_", "passage"],
        )
        .with_additional(["id vector"])
        .with_limit(8)
    )

    result = query.do()

    return result["data"]["Get"][class_name]


def test_index_weaviate_component(monkeypatch):
    client = weaviate.Client(embedded_options=EmbeddedOptions())

    pandas_df = pd.DataFrame(
        [
            ("Lorem ipsum dolor", np.array([1.0, 2.0])),
            ("ligula eget dolor", np.array([2.0, 3.0])),
        ],
        columns=["text", "embedding"],
    )
    dask_df = dd.from_pandas(pandas_df, npartitions=2)

    index_component = IndexWeaviateComponent(
        weaviate_url="http://localhost:6666",  # local weaviate instance running on port 6666
        batch_size=10,
        dynamic=True,
        num_workers=2,
        overwrite=True,
        class_name="TestClass",
        vectorizer=None,
    )

    index_component.write(dask_df)
    written_objects = get_written_objects(client, "TestClass")

    for _object in written_objects:
        matching_row = pandas_df.loc[int(_object["id_"])]
        assert matching_row.text == _object["passage"]
        assert matching_row.embedding.tolist() == _object["_additional"]["vector"]
