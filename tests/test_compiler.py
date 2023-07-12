import datetime
from pathlib import Path

import pytest
import yaml
from fondant.compiler import DockerCompiler
from fondant.exceptions import InvalidPipelineExecution
from fondant.pipeline import ComponentOp, Pipeline

COMPONENTS_PATH = Path("./tests/example_pipelines/valid_pipeline")
VALID_DOCKER_PIPELINE = Path("./tests/example_pipelines/compiled_pipeline/")
MOCK_BASE_PATH = Path("./tests/example_pipelines/mock_base_path/")

TEST_PIPELINES = [
    (
        "example_1",
        {
            "components": [
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
                ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "fourth_component.yaml"),
                    arguments={
                        "storage_args": "a dummy string arg",
                        "some_list": [1, 2, 3],
                    },
                ),
            ],
            "valid_resume_component": "second_component",
            "invalid_resume_component": "fourth_component",
        },
    ),
    (
        "example_2",
        {
            "components": [
                ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "first_component.yaml"),
                    arguments={"storage_args": "a dummy string arg"},
                ),
                ComponentOp.from_registry(
                    name="image_cropping",
                    arguments={"cropping_threshold": 0, "padding": 0},
                ),
            ],
            "valid_resume_component": "image_cropping",
            "invalid_resume_component": "none_existing_component",
        },
    ),
]


@pytest.fixture()
def _freeze_time(monkeypatch):
    class FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls):
            return datetime.datetime(2023, 1, 1)

    monkeypatch.setattr(
        datetime,
        "datetime",
        FrozenDatetime,
    )


@pytest.fixture(params=TEST_PIPELINES)
def setup_pipeline(request, tmp_path, monkeypatch):
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    example_dir, component_dict = request.param

    prev_comp = None
    for component in component_dict["components"]:
        pipeline.add_op(component, dependencies=prev_comp)
        prev_comp = component

    pipeline.compile()

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", str(tmp_path / "test_pipeline.tgz"))

    return (
        example_dir,
        pipeline,
        component_dict["valid_resume_component"],
        component_dict["invalid_resume_component"],
    )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to docker-compose."""
    example_dir, pipeline, _, _ = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        with open(output_path) as src, open(
            VALID_DOCKER_PIPELINE / example_dir / "docker-compose.yml",
        ) as truth:
            assert yaml.safe_load(src) == yaml.safe_load(truth)


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler_partial_without_resume_component(
    setup_pipeline,
    tmp_path_factory,
):
    """Test compiling a partially executed pipeline."""
    example_dir, pipeline, _, _ = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        pipeline.base_path = MOCK_BASE_PATH
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(
            pipeline=pipeline,
            output_path=output_path,
            run_id=f"pipeline_run_{example_dir}",
        )

        with open(output_path) as src, open(
            VALID_DOCKER_PIPELINE / example_dir / "docker-compose-partial.yml",
        ) as truth:
            assert (
                yaml.safe_load(src)["services"].keys()
                == yaml.safe_load(truth)["services"].keys()
            )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler_partial_with_valid_resume_component(
    setup_pipeline,
    tmp_path_factory,
):
    """Test compiling a partially executed pipeline when a component to resume from is specified."""
    example_dir, pipeline, valid_resume_component, _ = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        pipeline.base_path = MOCK_BASE_PATH
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(
            pipeline=pipeline,
            output_path=output_path,
            run_id=f"pipeline_run_{example_dir}",
            resume_component=valid_resume_component,
        )
        with open(output_path) as src:
            spec = yaml.safe_load(src)
        assert spec["services"][valid_resume_component]["depends_on"] == {}


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler_partial_with_invalid_resume_component(
    setup_pipeline,
    tmp_path_factory,
):
    """Test that a valid error is returned when attempting to resume a pipeline execution from an
    invalid specified component.
    """
    example_dir, pipeline, _, invalid_resume_component = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        pipeline.base_path = MOCK_BASE_PATH
        output_path = str(fn / "docker-compose.yml")
        with pytest.raises(InvalidPipelineExecution):
            compiler.compile(
                pipeline=pipeline,
                output_path=output_path,
                run_id=f"pipeline_run_{example_dir}",
                resume_component=invalid_resume_component,
            )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_local_path(setup_pipeline, tmp_path_factory):
    """Test that a local path is applied correctly as a volume and in the arguments."""
    # volumes are only created for local existing directories
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, pipeline, _, _ = setup_pipeline
        work_dir = f"/{fn.stem}"
        pipeline.base_path = str(fn)
        compiler = DockerCompiler()
        compiler.compile(pipeline=pipeline, output_path=fn / "docker-compose.yml")

        # read the generated docker-compose file
        with open(fn / "docker-compose.yml") as f_spec:
            spec = yaml.safe_load(f_spec)

        for name, service in spec["services"].items():
            # check if volumes are defined correctly
            assert service["volumes"] == [
                {
                    "source": str(fn),
                    "target": work_dir,
                    "type": "bind",
                },
            ]
            # check if commands are patched to use the working dir
            commands_with_dir = [
                f"{work_dir}/{name}/test_pipeline-20230101000000/manifest.json",
                f'{{"run_id": "test_pipeline-20230101000000", "base_path": "{work_dir}"}}',
            ]
            for command in commands_with_dir:
                assert command in service["command"]


@pytest.mark.usefixtures("_freeze_time")
def test_docker_remote_path(setup_pipeline, tmp_path_factory):
    """Test that a remote path is applied correctly in the arguments and no volume."""
    _, pipeline, _, __ = setup_pipeline
    remote_dir = "gs://somebucket/artifacts"
    pipeline.base_path = remote_dir
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        compiler.compile(pipeline=pipeline, output_path=fn / "docker-compose.yml")

        # read the generated docker-compose file
        with open(fn / "docker-compose.yml") as f_spec:
            spec = yaml.safe_load(f_spec)

        for name, service in spec["services"].items():
            # check that no volumes are created
            assert service["volumes"] == []
            # check if commands are patched to use the remote dir
            commands_with_dir = [
                f"{remote_dir}/{name}/test_pipeline-20230101000000/manifest.json",
                f'{{"run_id": "test_pipeline-20230101000000", "base_path": "{remote_dir}"}}',
            ]
            for command in commands_with_dir:
                assert command in service["command"]


@pytest.mark.usefixtures("_freeze_time")
def test_docker_extra_volumes(setup_pipeline, tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, pipeline, _, _ = setup_pipeline
        pipeline.base_path = str(fn)
        compiler = DockerCompiler()
        # define some extra volumes to be mounted
        extra_volumes = ["hello:there", "general:kenobi"]
        compiler.compile(
            pipeline=pipeline,
            output_path=fn / "docker-compose.yml",
            extra_volumes=extra_volumes,
        )

        # read the generated docker-compose file
        with open(fn / "docker-compose.yml") as f_spec:
            spec = yaml.safe_load(f_spec)
        for _name, service in spec["services"].items():
            assert all(
                extra_volume in service["volumes"] for extra_volume in extra_volumes
            )
