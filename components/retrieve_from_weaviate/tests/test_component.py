import tempfile

import numpy as np
import pandas as pd
import weaviate
from weaviate.embedded import EmbeddedOptions

from src.main import RetrieveFromWeaviateComponent


def set_up_instance(client):
    """Set up an embedded instance using the provided client."""
    data = [
        {
            "data_object": {
                "passage": "foo",
            },
            "vector": np.array([1.0, 2.0]),
        },
        {
            "data_object": {
                "passage": "bar",
            },
            "vector": np.array([2.0, 3.0]),
        },
    ]

    for entry in data:
        client.data_object.create(
            entry["data_object"],
            class_name="Test",
            vector=entry["vector"],
        )

    return "http://localhost:6666"


def test_component():
    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["1", "2"],
            "embedding": [np.array([1.0, 2.0]), np.array([2.0, 3.0])],
        },
    )
    input_dataframe = input_dataframe.set_index("id")

    expected_output_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["1", "2"],
            "retrieved_chunks": [["foo", "bar"], ["bar", "foo"]],
        },
    )
    expected_output_dataframe = expected_output_dataframe.set_index("id")

    with tempfile.TemporaryDirectory() as tmpdir:
        client = weaviate.Client(
            embedded_options=EmbeddedOptions(
                persistence_data_path=tmpdir,
            ),
        )
        url = set_up_instance(client)

        component = RetrieveFromWeaviateComponent(
            weaviate_url=url,
            class_name="Test",
            top_k=2,
            additional_config={},
            additional_headers={},
            hybrid_query=None,
            hybrid_alpha=None,
        )

        output_dataframe = component.transform(input_dataframe)

    pd.testing.assert_frame_equal(
        left=expected_output_dataframe,
        right=output_dataframe["retrieved_chunks"].to_frame(),
        check_dtype=False,
    )
