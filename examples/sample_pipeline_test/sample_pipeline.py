# This file contains a sample pipeline. Loading data from a parquet file,
# using the load_from_parquet component, chain a custom dummy component, and use
# the reusable chunking component
import os
import pyarrow as pa
from pathlib import Path
from fondant.pipeline import Pipeline

os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"
BASE_PATH = Path("./examples/sample_pipeline_test")
Path(BASE_PATH / ".artifacts").mkdir(parents=True, exist_ok=True)
NUMBER_OF_COMPONENTS = 3

# Define pipeline
pipeline = Pipeline(name="dummy-pipeline", base_path=str(BASE_PATH / ".artifacts"))

# Load from hub component
load_component_column_mapping = {
    "text": "text_data",
}

dataset = pipeline.read(
    name_or_path="load_from_parquet",
    arguments={
        "dataset_uri": "/data/sample.parquet",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 5,
    },
    produces={"text_data": pa.string()},
)

dataset = dataset.apply(
    name_or_path=Path(BASE_PATH / "components" / "dummy_component"),
)

dataset.apply(
    name_or_path="chunk_text",
    arguments={"chunk_size": 10, "chunk_overlap": 2},
    consumes={"text": "text_data"},
)
