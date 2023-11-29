import typing as t

import pandas as pd

from src.main import LAIONRetrievalComponent


def test_component(monkeypatch):
    def mocked_client_query(text: str) -> t.List[dict]:
        if text == "first prompt":
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
        if text == "second prompt":
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
        msg = f"Unexpected value: `text` was {text}"
        raise ValueError(msg)

    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["1", "2"],
            "prompt": ["first prompt", "second prompt"],
        },
    )

    expected_output_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["a", "b", "c", "d"],
            "image_url": ["http://a", "http://b", "http://c", "http://d"],
            "prompt_id": ["1", "1", "2", "2"],
        },
    )
    expected_output_dataframe = expected_output_dataframe.set_index("id")

    component = LAIONRetrievalComponent(
        num_images=2,
        aesthetic_score=9,
        aesthetic_weight=0.5,
        url="",
    )

    monkeypatch.setattr(component.client, "query", mocked_client_query)

    output_dataframe = component.transform(input_dataframe)

    pd.testing.assert_frame_equal(
        left=expected_output_dataframe,
        right=output_dataframe,
        check_dtype=False,
    )
