import typing as t

import numpy as np
import pandas as pd

from src.main import LAIONRetrievalComponent


def test_component(monkeypatch):
    def mocked_client_query(embedding_input: t.List[float]) -> t.List[dict]:
        if embedding_input == [1, 2]:
            return [
                {
                    "id": "a",
                    "url": "http://a",
                },
                {
                    "id": "b",
                    "url": "http://b",
                },
            ]
        if embedding_input == [2, 3]:
            return [
                {
                    "id": "c",
                    "url": "http://c",
                },
                {
                    "id": "d",
                    "url": "http://d",
                },
            ]
        msg = f"Unexpected value: `embeddings_input` was {embedding_input}"
        raise ValueError(msg)

    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["1", "2"],
            "embedding": [np.array([1, 2]), np.array([2, 3])],
        },
    )

    expected_output_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["a", "b", "c", "d"],
            "url": ["http://a", "http://b", "http://c", "http://d"],
            "embedding_id": ["1", "1", "2", "2"],
        },
    )
    expected_output_dataframe = expected_output_dataframe.set_index("id")

    component = LAIONRetrievalComponent(
        num_images=2,
        aesthetic_score=9,
        aesthetic_weight=0.5,
    )

    monkeypatch.setattr(component.client, "query", mocked_client_query)

    output_dataframe = component.transform(input_dataframe)

    pd.testing.assert_frame_equal(
        left=expected_output_dataframe,
        right=output_dataframe,
        check_dtype=False,
    )
