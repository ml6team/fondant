# This file contains a sample pipeline. Loading data from a parquet file,
# using the load_from_parquet component, chain a custom dummy component, and use
# the reusable chunking component
import glob
import logging
import os
from pathlib import Path

import pytest
from fondant.pipeline import ComponentOp, Pipeline
from fondant.pipeline.compiler import DockerCompiler
from fondant.pipeline.runner import DockerRunner

logger = logging.getLogger(__name__)

# TODO: probably removable after we have solved #344
# work around to make test executable on M1 Macbooks
os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"

BASE_PATH = Path("./tests/integration_tests/sample_pipeline_test")
NUMBER_OF_COMPONENTS = 3


@pytest.fixture()
def sample_pipeline(data_dir="./data") -> Pipeline:
    # Define pipeline
    pipeline = Pipeline(pipeline_name="dummy-pipeline", base_path=data_dir)

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


@pytest.mark.skip(reason="Skipping due to random failure.")
def test_local_runner(sample_pipeline, tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as data_dir:
        sample_pipeline.base_path = str(data_dir)
        DockerCompiler().compile(
            sample_pipeline,
            output_path="docker-compose.yaml",
            extra_volumes=[
                str(Path("tests/integration_tests/sample_pipeline_test/data").resolve())
                + ":/data",
            ],
        )
        DockerRunner().run("docker-compose.yaml")

        assert os.path.exists(data_dir / "dummy-pipeline")
        assert os.path.exists(data_dir / "dummy-pipeline" / "cache")
        pipeline_dirs = glob.glob(
            str(data_dir / "dummy-pipeline" / "dummy-pipeline-*" / "*"),
        )

        assert len(pipeline_dirs) == NUMBER_OF_COMPONENTS
        for dir in pipeline_dirs:
            assert os.path.exists(Path(dir) / "index")
            assert os.path.exists(Path(dir) / "text")
            assert os.path.exists(Path(dir) / "manifest.json")
