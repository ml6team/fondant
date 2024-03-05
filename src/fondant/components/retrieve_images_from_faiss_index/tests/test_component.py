import dask.dataframe as dd
import pandas as pd

from src.main import RetrieveImagesFromFaissIndex


def test_component():
    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "prompt": ["country style kitchen", "cozy living room"],
        },
    )

    input_dataframe = input_dataframe.set_index("id")
    input_dataframe = input_dataframe
    input_dataframe = dd.from_pandas(input_dataframe, npartitions=2)
    input_dataframe["prompt"] = input_dataframe["prompt"].astype(str)

    # Run component
    component = RetrieveImagesFromFaissIndex(
        url_mapping_path="gs://soy-audio-379412-embed-datacomp/12M/id_mapping",
        faiss_index_path="gs://soy-audio-379412-embed-datacomp/12M/faiss",
    )

    component.setup()
    output_dataframe = component.transform(input_dataframe)
    assert output_dataframe.columns.tolist() == [
        "prompt_id",
        "prompt",
        "image_index",
        "image_url",
    ]
