import dask.dataframe as dd
import pandas as pd

from src.main import RetrieveImagesFromFaissIndex


def test_component():
    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["1", "2"],
            "prompt": ["first prompt", "second prompt"],
        },
    )

    input_dataframe = input_dataframe.set_index("id")

    pd.DataFrame.from_dict(
        {
            "id": ["a", "b", "c", "d"],
            "image_url": ["http://a", "http://b", "http://c", "http://d"],
            "prompt_id": ["1", "1", "2", "2"],
        },
    )

    component = RetrieveImagesFromFaissIndex(
        dataset_url="./tests/resources",
    )

    input_dataframe = dd.from_pandas(input_dataframe, npartitions=4)
    output_dataframe = component.transform(input_dataframe)
    assert output_dataframe.columns.tolist() == [
        "prompt_id",
        "prompt",
        "image_index",
        "image_url",
    ]
