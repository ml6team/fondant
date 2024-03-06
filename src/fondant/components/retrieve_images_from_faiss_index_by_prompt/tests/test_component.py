import pandas as pd

from src.main import RetrieveImagesFromFaissIndexByPrompt


def test_component():
    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "prompt": ["country style kitchen", "cozy living room"],
        },
    )

    input_dataframe = input_dataframe.set_index("id")
    input_dataframe["prompt"] = input_dataframe["prompt"].astype(str)

    # Run component
    component = RetrieveImagesFromFaissIndexByPrompt(
        url_mapping_path="gs://soy-audio-379412-embed-datacomp/12M/id_mapping",
        faiss_index_path="gs://soy-audio-379412-embed-datacomp/12M/faiss",
    )

    component.setup()
    output_dataframe = component.transform(input_dataframe)
    assert output_dataframe.columns.tolist() == [
        "prompt_id",
        "image_url",
    ]
