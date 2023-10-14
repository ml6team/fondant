"""Unit test for generate embeddings component."""
import json
from math import isclose

import pandas as pd

from components.generate_embeddings.src.main import GenerateEmbeddingsComponent


def embeddings_close(a, b):
    return all(isclose(x, y, abs_tol=1e-5) for x, y in zip(a, b))


def test_run_component_test():
    """Test generate embeddings component."""
    with open("lorem_300.txt", encoding="utf-8") as f:
        lorem_300 = f.read()
    with open("lorem_400.txt", encoding="utf-8") as f:
        lorem_400 = f.read()

    # Given: Dataframe with text
    data = [
        {"data": "Hello World!!"},
        {"data": lorem_300},
        {"data": lorem_400},
    ]

    DATA_LENTGH = 3

    dataframe = pd.concat({"text": pd.DataFrame(data)}, axis=1, names=["text", "data"])

    component = GenerateEmbeddingsComponent(
        model_provider="huggingface",
        model="all-MiniLM-L6-v2",
        api_keys={},
    )

    dataframe = component.transform(dataframe=dataframe)

    with open("hello_world_embedding.txt", encoding="utf-8") as f:
        hello_world_embedding = f.read()
        HELLO_WORLD_EMBEDDING = json.loads(hello_world_embedding)

    # Then: right embeddings are generated
    assert len(dataframe) == DATA_LENTGH
    assert embeddings_close(
        dataframe.iloc[0]["text"]["embedding"],
        HELLO_WORLD_EMBEDDING,
    )
    # Then: too long text is truncated and thus embeddings are the same
    assert (
        dataframe.iloc[1]["text"]["embedding"] == dataframe.iloc[2]["text"]["embedding"]
    )
