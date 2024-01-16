# This file contains a sample pipeline. Loading data from a parquet file,
# using the load_from_parquet component, chain a custom dummy component, and use
# the reusable chunking component
from pathlib import Path

import pyarrow as pa

from fondant.pipeline import Pipeline

BASE_PATH = Path("./.artifacts").resolve()
BASE_PATH.mkdir(parents=True, exist_ok=True)

# Define pipeline
pipeline = Pipeline(name="dummy-pipeline", base_path=str(BASE_PATH))

# Load from hub component
load_component_column_mapping = {
    "text": "text_data",
}

dataset = pipeline.read(
    "load_from_parquet",
    arguments={
        "dataset_uri": "/data/sample.parquet",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 5,
    },
    produces={"text_data": pa.string()},
)

dataset = dataset.apply(
    "./components/dummy_component",
)

dataset.apply(
    "chunk_text",
    arguments={"chunk_size": 10, "chunk_overlap": 2},
    consumes={"text": "text_data"},
)
