"""Unit test for generate embeddings component."""
import json
from math import isclose

import pandas as pd

from src.main import EmbedTextComponent


def embeddings_close(a, b):
    return all(isclose(x, y, abs_tol=1e-5) for x, y in zip(a, b))


def test_run_component_test():
    """Test generate embeddings component."""
    with open("tests/lorem_300.txt", encoding="utf-8") as f:
        lorem_300 = f.read()
    with open("tests/lorem_400.txt", encoding="utf-8") as f:
        lorem_400 = f.read()

    # Given: Dataframe with text
    data = [
        {"text": "Hello World!!"},
        {"text": lorem_300},
        {"text": lorem_400},
    ]

    DATA_LENTGH = 3

    dataframe = pd.DataFrame(data)

    component = EmbedTextComponent(
        model_provider="huggingface",
        model="all-MiniLM-L6-v2",
        api_keys={},
        auth_kwargs={},
        tries=5,
    )

    dataframe = component.transform(dataframe=dataframe)

    with open("tests/hello_world_embedding.txt", encoding="utf-8") as f:
        hello_world_embedding = f.read()
        hello_world_embedding = json.loads(hello_world_embedding)

    # Then: right embeddings are generated
    assert len(dataframe) == DATA_LENTGH
    assert embeddings_close(
        dataframe.iloc[0]["embedding"],
        hello_world_embedding,
    )
    # Then: too long text is truncated and thus embeddings are the same
    assert dataframe.iloc[1]["embedding"] == dataframe.iloc[2]["embedding"]
