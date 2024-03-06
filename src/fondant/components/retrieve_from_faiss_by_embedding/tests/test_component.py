import numpy as np
import pandas as pd
from main import RetrieveFromFaissByEmbedding


def test_component_from_embeddings():
    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "embedding": [
                np.random.rand(512, 512),  # noqa NPY002
                np.random.rand(512, 512),  # noqa NPY002
            ],
        },
    )

    input_dataframe = input_dataframe.set_index("id")

    # Run component
    component = RetrieveFromFaissByEmbedding(
        url_mapping_path="hf://datasets/fondant-ai/datacomp-small-clip/id_mapping/id_mapping",
        faiss_index_path="hf://datasets/fondant-ai/datacomp-small-clip/faiss",
    )

    component.setup()
    output_dataframe = component.transform(input_dataframe)
    assert output_dataframe.columns.tolist() == [
        "image_url",
    ]
