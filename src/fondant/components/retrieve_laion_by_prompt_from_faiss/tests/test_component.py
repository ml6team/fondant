import pandas as pd
from main import RetrieveFromLaionByPrompt


def test_component(monkeypatch):
    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["1", "2"],
            "prompt": ["first prompt", "second prompt"],
        },
    )

    input_dataframe = input_dataframe.set_index("id")

    expected_output_dataframe = pd.DataFrame.from_dict(
        {
            "id": ["a", "b", "c", "d"],
            "image_url": ["http://a", "http://b", "http://c", "http://d"],
            "prompt_id": ["1", "1", "2", "2"],
        },
    )
    expected_output_dataframe = expected_output_dataframe.set_index("id")

    component = RetrieveFromLaionByPrompt(
        dataset_url="gs://soy-audio-379412-embed-datacomp/estimate/index-datacomp-small-64/index-datacomp-small-64-20240215120946/id_mapping",
        index_url="./faiss-idx",
    )

    output_dataframe = component.transform(input_dataframe)

    print(output_dataframe.head())
