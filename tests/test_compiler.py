import datetime
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml
from fondant.compiler import DockerCompiler, KubeFlowCompiler
from fondant.pipeline import ComponentOp, Pipeline

COMPONENTS_PATH = Path("./tests/example_pipelines/valid_pipeline")

VALID_PIPELINE = Path("./tests/example_pipelines/compiled_pipeline/")

TEST_PIPELINES = [
    (
        "example_1",
        [
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "first_component"),
                    arguments={"storage_args": "a dummy string arg"},
                    input_partition_rows="disable",
                    number_of_gpus=1,
                    preemptible=True,
                ),
                "cache_key": "1",
            },
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "second_component"),
                    arguments={"storage_args": "a dummy string arg"},
                    input_partition_rows="10",
                ),
                "cache_key": "2",
            },
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "third_component"),
                    arguments={
                        "storage_args": "a dummy string arg",
                    },
                ),
                "cache_key": "3",
            },
        ],
    ),
    (
        "example_2",
        [
            {
                "component_op": ComponentOp(
                    Path(COMPONENTS_PATH / "example_1" / "first_component"),
                    arguments={"storage_args": "a dummy string arg"},
                ),
                "cache_key": "1",
            },
            {
                "component_op": ComponentOp.from_registry(
                    name="image_cropping",
                    arguments={"cropping_threshold": 0, "padding": 0},
                ),
                "cache_key": "2",
            },
        ],
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
    example_dir, components = request.param
    prev_comp = None
    cache_dict = {}
    for component_dict in components:
        component = component_dict["component_op"]
        cache_key = component_dict["cache_key"]
        # set the cache_key as a default argument in the lambda function to avoid setting attribute
        # by reference
        monkeypatch.setattr(
            component,
            "get_component_cache_key",
            lambda cache_key=cache_key: cache_key,
        )
        pipeline.add_op(component, dependencies=prev_comp)
        prev_comp = component
        cache_dict[component.name] = cache_key

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", str(tmp_path / "test_pipeline.tgz"))

    return example_dir, pipeline, cache_dict


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to docker-compose."""
    example_dir, pipeline, _ = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path, build_args=[])
        with open(output_path) as src, open(
            VALID_PIPELINE / example_dir / "docker-compose.yml",
        ) as truth:
            assert yaml.safe_load(src) == yaml.safe_load(truth)


@pytest.mark.usefixtures("_freeze_time")
def test_docker_local_path(setup_pipeline, tmp_path_factory):
    """Test that a local path is applied correctly as a volume and in the arguments."""
    # volumes are only created for local existing directories
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, pipeline, cache_dict = setup_pipeline
        work_dir = f"/{fn.stem}"
        pipeline.base_path = str(fn)
        compiler = DockerCompiler()
        compiler.compile(pipeline=pipeline, output_path=fn / "docker-compose.yml")

        # read the generated docker-compose file
        with open(fn / "docker-compose.yml") as f_spec:
            spec = yaml.safe_load(f_spec)

        expected_run_id = "test_pipeline-20230101000000"
        for name, service in spec["services"].items():
            # check if volumes are defined correctly

            cache_key = cache_dict[name]
            assert service["volumes"] == [
                {
                    "source": str(fn),
                    "target": work_dir,
                    "type": "bind",
                },
            ]
            # check if commands are patched to use the working dir
            commands_with_dir = [
                f"{work_dir}/{pipeline.name}/{expected_run_id}/{name}/manifest.json",
                f'{{"base_path": "{work_dir}", "pipeline_name": "{pipeline.name}",'
                f' "run_id": "{expected_run_id}", "component_id": "{name}",'
                f' "cache_key": "{cache_key}"}}',
            ]
            for command in commands_with_dir:
                assert command in service["command"]


@pytest.mark.usefixtures("_freeze_time")
def test_docker_remote_path(setup_pipeline, tmp_path_factory):
    """Test that a remote path is applied correctly in the arguments and no volume."""
    _, pipeline, cache_dict = setup_pipeline
    remote_dir = "gs://somebucket/artifacts"
    pipeline.base_path = remote_dir
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        compiler.compile(pipeline=pipeline, output_path=fn / "docker-compose.yml")

        # read the generated docker-compose file
        with open(fn / "docker-compose.yml") as f_spec:
            spec = yaml.safe_load(f_spec)

        expected_run_id = "test_pipeline-20230101000000"
        for name, service in spec["services"].items():
            cache_key = cache_dict[name]
            # check that no volumes are created
            assert service["volumes"] == []
            # check if commands are patched to use the remote dir
            commands_with_dir = [
                f"{remote_dir}/{pipeline.name}/{expected_run_id}/{name}/manifest.json",
                f'{{"base_path": "{remote_dir}", "pipeline_name": "{pipeline.name}",'
                f' "run_id": "{expected_run_id}", "component_id": "{name}",'
                f' "cache_key": "{cache_key}"}}',
            ]
            for command in commands_with_dir:
                assert command in service["command"]


@pytest.mark.usefixtures("_freeze_time")
def test_docker_extra_volumes(setup_pipeline, tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, pipeline, _ = setup_pipeline
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


@pytest.mark.usefixtures("_freeze_time")
def test_kubeflow_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to kubeflow."""
    example_dir, pipeline, _ = setup_pipeline
    compiler = KubeFlowCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        with open(output_path) as src, open(
            VALID_PIPELINE / example_dir / "kubeflow_pipeline.yml",
        ) as truth:
            assert yaml.safe_load(src) == yaml.safe_load(truth)


@pytest.mark.usefixtures("_freeze_time")
def test_kubeflow_configuration(tmp_path_factory):
    """Test that the kubeflow pipeline can be configured."""
    pipeline = Pipeline(
        pipeline_name="test_pipeline",
        pipeline_description="description of the test pipeline",
        base_path="/foo/bar",
    )
    component_1 = ComponentOp(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        node_pool_name="a_node_pool",
        node_pool_label="a_node_pool_label",
        number_of_gpus=1,
    )
    pipeline.add_op(component_1)
    compiler = KubeFlowCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(pipeline=pipeline, output_path=output_path)
        with open(output_path) as src, open(
            VALID_PIPELINE / "kubeflow_pipeline.yml",
        ) as truth:
            assert yaml.safe_load(src) == yaml.safe_load(truth)


def test_kfp_import():
    """Test that the kfp import throws the correct error."""
    with mock.patch.dict(sys.modules):
        # remove kfp from the modules
        sys.modules["kfp"] = None
        with pytest.raises(ImportError):
            _ = KubeFlowCompiler()
