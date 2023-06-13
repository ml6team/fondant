from pathlib import Path

import pytest

from fondant.compiler import DockerCompiler
from fondant.pipeline import ComponentOp, Pipeline

COMPONENTS_PATH = Path(__file__).parent / "example_pipelines/valid_pipeline"

VALID_DOCKER_PIPELINE = (
    Path(__file__).parent / "example_pipelines/compiled_pipeline/docker-compose.yml"
)

TEST_PIPELINES = [
    (
        "example_1",
        ["first_component.yaml", "second_component.yaml", "third_component.yaml"],
    ),
]


@pytest.fixture(params=TEST_PIPELINES)
def pipeline(request, tmp_path, monkeypatch):
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    example_dir, component_specs = request.param

    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(COMPONENTS_PATH / example_dir)

    prev_comp = None
    for component_spec in component_specs:
        component_op = ComponentOp(
            Path(components_path / component_spec), arguments=component_args
        )
        pipeline.add_op(component_op, dependencies=prev_comp)
        prev_comp = component_op

    pipeline.compile()

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", str(tmp_path / "test_pipeline.tgz"))

    return pipeline


def test_docker_compiler(pipeline, tmp_path_factory):
    """Test compiling a pipeline to docker-compose."""
    compiler = DockerCompiler(pipeline=pipeline)
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(package_path=output_path)
        with open(output_path, "r") as src, open(VALID_DOCKER_PIPELINE, "r") as truth:
            assert src.read() == truth.read()


def test_docker_local_path(pipeline, tmp_path_factory):
    """Test that a local path is applied correctly as a volume and in the arguments."""
    # volumes are only create for local existing directories
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        work_dir = f"/{fn.stem}"
        pipeline.base_path = str(fn)
        compiler = DockerCompiler(pipeline=pipeline)
        compiler._patch_path()
        assert compiler.path == work_dir
        spec = compiler._generate_spec()
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


def test_docker_remote_path(pipeline):
    """Test that a remote path is applied correctly in the arguments and no volume."""
    remote_dir = "gs://somebucket/artifacts"
    pipeline.base_path = remote_dir
    compiler = DockerCompiler(pipeline=pipeline)
    compiler._patch_path()
    assert compiler.path == remote_dir
    spec = compiler._generate_spec()
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
