# This file contains a sample pipeline. Loading data from a parquet file, using the load_from_parquet
# component, applies text normalisation and transform the text to upper case (using a custom
# dummy component).
import glob
import logging
import os
import shutil
from pathlib import Path

import pytest
from fondant.pipeline import ComponentOp, Pipeline
from fondant.pipeline.compiler import DockerCompiler
from fondant.pipeline.runner import DockerRunner

logger = logging.getLogger(__name__)

# TODO: probably removable after we have solved #344
# work around to make test executable on M1 Macbooks
os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"

BASE_PATH = Path("./tests/sample_pipeline_test")
DATA_DIR = Path(BASE_PATH / "data")
NUMBER_OF_COMPONENTS = 3


@pytest.fixture()
def sample_pipeline() -> Pipeline:
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

    custom_dummy_component = ComponentOp(
        component_dir=Path(BASE_PATH / "components" / "dummy_component"),
    )

    chunk_text = ComponentOp.from_registry(
        name="chunk_text",
        arguments={"chunk_size": 10, "chunk_overlap": 2},
    )

    # Add components to the pipeline
    pipeline.add_op(load_from_file)
    pipeline.add_op(custom_dummy_component, dependencies=load_from_file)
    pipeline.add_op(chunk_text, dependencies=[custom_dummy_component])

    return pipeline


def test_local_runner(sample_pipeline):
    DockerCompiler().compile(
        sample_pipeline,
        output_path="docker-compose.yaml",
        extra_volumes=[str(DATA_DIR.resolve()) + ":/data"],
    )
    DockerRunner().run("docker-compose.yaml")

    assert os.path.exists(DATA_DIR / "dummy-pipeline")
    assert os.path.exists(DATA_DIR / "dummy-pipeline" / "cache")
    pipeline_dirs = glob.glob(
        str(DATA_DIR / "dummy-pipeline" / "dummy-pipeline-*" / "*"),
    )

    assert len(pipeline_dirs) == NUMBER_OF_COMPONENTS
    for dir in pipeline_dirs:
        assert os.path.exists(Path(dir) / "index")
        assert os.path.exists(Path(dir) / "text")
        assert os.path.exists(Path(dir) / "manifest.json")

    try:
        shutil.rmtree(DATA_DIR / "dummy-pipeline")
    except PermissionError:
        # No cleanup needed for the ci/cd pipeline as far as only one local runner test is executed.
        # A PermissionError will be thrown when this process tries to delete the folders which were
        # created from the docker environment.
        logger.info("PermissionError: Not able to delete the data folder.")
