from pathlib import Path

import pytest
import yaml

from fondant.compiler import DockerCompiler
from fondant.pipeline import ComponentOp, Pipeline

COMPONENTS_PATH = Path(__file__).parent / "example_pipelines/valid_pipeline"

VALID_DOCKER_PIPELINE = Path(__file__).parent / "example_pipelines/compiled_pipeline/"

TEST_PIPELINES = [
    (
        "example_1",
        [
            ComponentOp(
                Path(COMPONENTS_PATH / "example_1" / "first_component.yaml"),
                arguments={"storage_args": "a dummy string arg"},
            ),
            ComponentOp(
                Path(COMPONENTS_PATH / "example_1" / "second_component.yaml"),
                arguments={"storage_args": "a dummy string arg"},
            ),
            ComponentOp(
                Path(COMPONENTS_PATH / "example_1" / "third_component.yaml"),
                arguments={"storage_args": "a dummy string arg"},
            ),
        ],
    ),
    (
        "example_2",
        [
            ComponentOp(
                Path(COMPONENTS_PATH / "example_1" / "first_component.yaml"),
                arguments={"storage_args": "a dummy string arg"},
            ),
            ComponentOp.from_registry(
                name="image_cropping", arguments={"cropping_threshold": 0, "padding": 0}
            ),
        ],
    ),
]


@pytest.fixture(params=TEST_PIPELINES)
def setup_pipeline(request, tmp_path, monkeypatch):
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    example_dir, components = request.param

    prev_comp = None
    for component in components:
        pipeline.add_op(component, dependencies=prev_comp)
        prev_comp = component

    pipeline.compile()

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", str(tmp_path / "test_pipeline.tgz"))

    return (example_dir, pipeline)


def test_docker_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to docker-compose."""
    example_dir, pipeline = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        with open(output_path, "r") as src, open(
            VALID_DOCKER_PIPELINE / example_dir / "docker-compose.yml", "r"
        ) as truth:
            assert src.read() == truth.read()


def test_docker_local_path(setup_pipeline, tmp_path_factory):
    """Test that a local path is applied correctly as a volume and in the arguments."""
    # volumes are only create for local existing directories
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, pipeline = setup_pipeline
        work_dir = f"/{fn.stem}"
        pipeline.base_path = str(fn)
        compiler = DockerCompiler()
        compiler.compile(pipeline=pipeline, output_path=fn / "docker-compose.yml")

        # read the generated docker-compose file
        with open(fn / "docker-compose.yml") as f_spec:
            spec = yaml.safe_load(f_spec)

        for service in spec["services"].values():
            # check if volumes are defined correctly
            assert service["volumes"] == [
                {
                    "source": str(fn),
                    "target": work_dir,
                    "type": "bind",
                }
            ]
            # check if commands are patched to use the working dir
            commands_with_dir = [
                f"{work_dir}/manifest.txt",
                f'{{"run_id": "test_pipeline", "base_path": "{work_dir}"}}',
            ]
            for command in commands_with_dir:
                assert command in service["command"]


def test_docker_remote_path(setup_pipeline, tmp_path_factory):
    """Test that a remote path is applied correctly in the arguments and no volume."""
    _, pipeline = setup_pipeline
    remote_dir = "gs://somebucket/artifacts"
    pipeline.base_path = remote_dir
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        compiler.compile(pipeline=pipeline, output_path=fn / "docker-compose.yml")

        # read the generated docker-compose file
        with open(fn / "docker-compose.yml") as f_spec:
            spec = yaml.safe_load(f_spec)

        for service in spec["services"].values():
            # check that no volumes are created
            assert service["volumes"] == []
            # check if commands are patched to use the remote dir
            commands_with_dir = [
                f"{remote_dir}/manifest.txt",
                f'{{"run_id": "test_pipeline", "base_path": "{remote_dir}"}}',
            ]
            for command in commands_with_dir:
                assert command in service["command"]
