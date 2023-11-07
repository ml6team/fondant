# This file contains a sample pipeline. Loading data from a parquet file, using  load_from_parquet
# component, applies text normalisation and transform the text to upper case (using a custom
# dummy component).

import glob
import os
import shutil
from pathlib import Path

from fondant.pipeline import ComponentOp, Pipeline
from fondant.pipeline.compiler import DockerCompiler
from fondant.pipeline.runner import DockerRunner

# TODO: can probably removed after we have solved #344
os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"

BASE_PATH = Path("./tests/example_pipelines/executable_pipeline")
DATA_DIR = Path(BASE_PATH / "data")


def initialise_pipeline() -> Pipeline:
    # Define pipeline
    pipeline = Pipeline(pipeline_name="dummy-pipeline", base_path=str(DATA_DIR))

    # Load from hub component
    load_component_column_mapping = {
        "text": "text_data",
    }

    load_from_file = ComponentOp(
        component_dir=Path(BASE_PATH / "components" / "load_from_parquet"),
        arguments={
            "dataset_uri": "/data/sample.parquet",
            "column_name_mapping": load_component_column_mapping,
            "n_rows_to_load": 5,
        },
    )

    chunk_text = ComponentOp.from_registry(
        name="chunk_text",
        arguments={"chunk_size": 10, "chunk_overlap": 2},
    )

    custom_dummy_component = ComponentOp(
        component_dir=Path(BASE_PATH / "components" / "dummy_component"),
    )

    # Add components to the pipeline
    pipeline.add_op(load_from_file)
    pipeline.add_op(custom_dummy_component, dependencies=load_from_file)
    pipeline.add_op(chunk_text, dependencies=[custom_dummy_component])

    return pipeline


def test_local_runner():
    pipeline = initialise_pipeline()
    DockerCompiler().compile(
        pipeline,
        output_path="docker-compose.yaml",
        extra_volumes=[str(DATA_DIR.resolve()) + ":/data"],
    )
    DockerRunner().run("docker-compose.yaml")

    # Paths to the folders you want to check
    assert os.path.exists(DATA_DIR / "dummy-pipeline")
    assert os.path.exists(DATA_DIR / "dummy-pipeline" / "cache")
    pipeline_dirs = glob.glob(
        str(DATA_DIR / "dummy-pipeline" / "dummy-pipeline-*" / "*"),
    )

    for dir in pipeline_dirs:
        assert os.path.exists(Path(dir) / "index")
        assert os.path.exists(Path(dir) / "text")
        assert os.path.exists(Path(dir) / "manifest.json")

    # Delete dummy-pipeline folder
    shutil.rmtree(DATA_DIR / "dummy-pipeline")
