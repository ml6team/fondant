"""Unit test for minhash generation component."""
import pandas as pd

from src.main import MinHashGeneratorComponent


def test_run_component_test():
    """Test MinHash generation."""
    # Given: Dataframe with text, one duplicate in
    data = [
        {"text": "This is my first sentence"},
        {"text": "This is my first sentence"},
        {"text": "This is a different sentence"},
    ]

    dataframe = pd.DataFrame(data)

    component = MinHashGeneratorComponent(shingle_ngram_size=3)

    dataframe = component.transform(dataframe=dataframe)

    # Then: dataframe contain minhashes for each entry
    assert any(
        dataframe.loc[0]["minhash"] == dataframe.loc[1]["minhash"],
    )
    assert not any(
        dataframe.loc[0]["minhash"] == dataframe.loc[2]["minhash"],
    )
