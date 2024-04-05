import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from fondant.component import DaskLoadComponent
from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidDatasetDefinition
from fondant.core.manifest import Manifest, Metadata
from fondant.core.schema import CloudCredentialsMount
from fondant.dataset import (
    ComponentOp,
    Dataset,
    Resources,
    lightweight_component,
)
from fondant.dataset.compiler import (
    DockerCompiler,
    KubeFlowCompiler,
    KubeflowComponentSpec,
    SagemakerCompiler,
    VertexCompiler,
)
from fondant.testing import (
    DockerComposeConfigs,
    KubeflowPipelineConfigs,
    VertexPipelineConfigs,
)

COMPONENTS_PATH = Path("./tests/pipeline/examples/pipelines/valid_pipeline")

VALID_PIPELINE = Path("./tests/pipeline/examples/pipelines/compiled_pipeline/")

TEST_PIPELINES = [
    (
        "example_1",
        [
            {
                "component_op": ComponentOp.from_component_yaml(
                    Path(COMPONENTS_PATH / "example_1" / "first_component"),
                    arguments={"storage_args": "a dummy string arg"},
                    input_partition_rows=10,
                    resources=Resources(
                        memory_limit="512M",
                        memory_request="256M",
                    ),
                    produces={"images_data": pa.binary()},
                ),
                "cache_key": "1",
            },
            {
                "component_op": ComponentOp.from_component_yaml(
                    Path(COMPONENTS_PATH / "example_1" / "second_component"),
                    arguments={"storage_args": "a dummy string arg"},
                    input_partition_rows=10,
                ),
                "cache_key": "2",
            },
            {
                "component_op": ComponentOp.from_component_yaml(
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
                "component_op": ComponentOp.from_component_yaml(
                    Path(COMPONENTS_PATH / "example_1" / "first_component"),
                    arguments={"storage_args": "a dummy string arg"},
                    produces={"images_data": pa.binary()},
                ),
                "cache_key": "1",
            },
            {
                "component_op": ComponentOp.from_component_yaml(
                    "crop_images",
                    arguments={"cropping_threshold": 0, "padding": 0},
                ),
                "cache_key": "2",
            },
        ],
    ),
]

component_specs_path = Path("./tests/core/examples/component_specs")


@pytest.fixture()
def valid_kubeflow_schema() -> dict:
    with open(component_specs_path / "kubeflow_component.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def valid_fondant_schema() -> dict:
    with open(component_specs_path / "valid_component.yaml") as f:
        return yaml.safe_load(f)


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
    working_directory = "/foo/bar"

    run_id = Dataset.get_run_id("testpipeline")

    manifest = Manifest.create(
        dataset_name="testpipeline",
        run_id=run_id,
    )
    dataset = Dataset(manifest=manifest)
    cache_dict = {}
    example_dir, components = request.param
    for component_dict in components:
        component = component_dict["component_op"]
        cache_key = component_dict["cache_key"]
        # set the cache_key as a default argument in the lambda function to avoid setting attribute
        # by reference
        monkeypatch.setattr(
            component,
            "get_component_cache_key",
            lambda cache_key=cache_key, previous_component_cache=None: cache_key,
        )
        dataset = dataset._apply(component)
        cache_dict[component.component_name] = cache_key

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(
        dataset.__class__,
        "package_path",
        str(tmp_path / "test_pipeline.tgz"),
    )

    return example_dir, working_directory, dataset, cache_dict


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to docker-compose."""
    example_dir, _, dataset, _ = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        working_directory = str(fn)
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
            build_args=[],
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        assert pipeline_configs.dataset_name == dataset.name
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # Get expected component configs
            component = dataset._graph[component_name]
            component_op = component["operation"]

            # Check that the component configs are correct
            assert component_configs.dependencies == component["dependencies"]
            assert component_configs.memory_limit is None
            assert component_configs.memory_request is None
            assert component_configs.cpu_limit is None
            assert component_configs.cpu_request is None
            if component_configs.accelerators:
                assert (
                    component_configs.accelerators.number_of_accelerators
                    == component_op.accelerators.number_of_accelerators
                )
            if component_op.input_partition_rows is not None:
                assert (
                    int(component_configs.arguments["input_partition_rows"])
                    == component_op.input_partition_rows
                )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_local_path(setup_pipeline, tmp_path_factory):
    """Test that a local path is applied correctly as a volume and in the arguments."""
    # volumes are only created for local existing directories
    with tmp_path_factory.mktemp("temp") as fn:
        # this is the directory mounted in the container
        _, _, dataset, cache_dict = setup_pipeline
        work_dir_stem = f"/{fn.stem}"
        working_directory = str(fn)
        compiler = DockerCompiler()
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        expected_run_id = "testpipeline-20230101000000"
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # check if volumes are defined correctly

            cache_key = cache_dict[component_name]
            assert component_configs.volumes == [
                {
                    "source": str(fn),
                    "target": work_dir_stem,
                    "type": "bind",
                },
            ]
            cleaned_pipeline_name = dataset.name.replace("_", "")
            # check if commands are patched to use the working dir
            expected_output_manifest_path = (
                f"{work_dir_stem}/{cleaned_pipeline_name}/{expected_run_id}"
                f"/{component_name}/manifest.json"
            )

            expected_metadata = {
                "dataset_name": "testpipeline",
                "run_id": expected_run_id,
                "cache_key": cache_key,
                "component_id": component_name,
                "manifest_location": f"{working_directory}/{dataset.name}/"
                f"{expected_run_id}/{component_name}/manifest.json",
            }

            assert (
                component_configs.arguments["output_manifest_path"]
                == expected_output_manifest_path
            )
            assert (
                json.loads(component_configs.arguments["metadata"]) == expected_metadata
            )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_remote_path(setup_pipeline, tmp_path_factory):
    """Test that a remote path is applied correctly in the arguments and no volume."""
    _, _, dataset, cache_dict = setup_pipeline
    working_directory = "gs://somebucket/artifacts"
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        expected_run_id = "testpipeline-20230101000000"
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            cache_key = cache_dict[component_name]
            # check that no volumes are created
            assert component_configs.volumes == []
            # check if commands are patched to use the remote dir
            cleaned_pipeline_name = dataset.name.replace("_", "")

            expected_output_manifest_path = (
                f"{working_directory}/{cleaned_pipeline_name}/{expected_run_id}"
                f"/{component_name}/manifest.json"
            )

            expected_metadata = {
                "dataset_name": cleaned_pipeline_name,
                "run_id": expected_run_id,
                "cache_key": cache_key,
                "component_id": component_name,
                "manifest_location": f"{working_directory}/{dataset.name}/"
                f"{expected_run_id}/{component_name}/manifest.json",
            }

            assert (
                component_configs.arguments["output_manifest_path"]
                == expected_output_manifest_path
            )
            assert (
                json.loads(component_configs.arguments["metadata"]) == expected_metadata
            )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_extra_volumes(setup_pipeline, tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    for auth_provider in CloudCredentialsMount:
        extra_auth_volume = auth_provider.get_path()

        with tmp_path_factory.mktemp("temp") as fn:
            # this is the directory mounted in the container
            _, _, dataset, _ = setup_pipeline
            working_directory = str(fn)
            compiler = DockerCompiler()
            # define some extra volumes to be mounted
            extra_volumes = ["hello:there", "general:kenobi"]
            extra_volumes.append(extra_auth_volume)
            output_path = str(fn / "docker-compose.yml")

            compiler.compile(
                dataset=dataset,
                working_directory=working_directory,
                output_path=output_path,
                extra_volumes=extra_volumes,
                auth_provider=auth_provider,
            )

            pipeline_configs = DockerComposeConfigs.from_spec(output_path)
            for _, service in pipeline_configs.component_configs.items():
                assert all(
                    extra_volume in service.volumes for extra_volume in extra_volumes
                )


@pytest.mark.usefixtures("_freeze_time")
def test_docker_configuration(tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    dataset = Dataset.create(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="GPU",
        ),
        produces={"captions_data": pa.string()},
        dataset_name="test_pipeline",
    )

    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        working_directory = str(fn)
        output_path = str(fn / "docker-compose.yaml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        component_config = pipeline_configs.component_configs["first_component"]
        assert component_config.accelerators[0].type == "gpu"
        assert component_config.accelerators[0].number == 1


@pytest.mark.usefixtures("_freeze_time")
def test_invalid_docker_configuration(tmp_path_factory):
    """Test that a valid error is returned when an unknown accelerator is set."""
    dataset = Dataset.create(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="unknown resource",
        ),
        produces={"captions_data": pa.string()},
        dataset_name="test_pipeline",
    )

    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn, pytest.raises(  # noqa PT012
        InvalidDatasetDefinition,
    ):
        working_directory = str(fn)
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path="kubeflow_pipeline.yml",
        )


def test_kubeflow_component_creation(valid_fondant_schema, valid_kubeflow_schema):
    """Test that the created kubeflow component matches the expected kubeflow component."""
    fondant_component = ComponentSpec.from_dict(valid_fondant_schema)
    kubeflow_component = KubeflowComponentSpec.from_fondant_component_spec(
        fondant_component,
        command=["fondant", "execute", "main"],
        image_uri="example_component:latest",
    )
    assert kubeflow_component._specification == valid_kubeflow_schema


def test_kubeflow_component_spec_to_file(valid_kubeflow_schema):
    """Test that the KubeflowComponentSpec can be written to a file."""
    kubeflow_component_spec = KubeflowComponentSpec(valid_kubeflow_schema)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "kubeflow_component_spec.yaml")
        kubeflow_component_spec.to_file(file_path)

        with open(file_path) as f:
            written_data = yaml.safe_load(f)

        # check if the written data is the same as the original data
        assert written_data == valid_kubeflow_schema


def test_kubeflow_component_spec_repr(valid_kubeflow_schema):
    """Test that the __repr__ method of KubeflowComponentSpec returns the expected string."""
    kubeflow_component_spec = KubeflowComponentSpec(valid_kubeflow_schema)
    expected_repr = f"KubeflowComponentSpec({valid_kubeflow_schema!r})"
    assert repr(kubeflow_component_spec) == expected_repr


@pytest.mark.usefixtures("_freeze_time")
def test_kubeflow_component_spec_from_lightweight_component(
    tmp_path_factory,
):
    @lightweight_component(
        base_image="python:3.10-slim-buster",
        extra_requires=["pandas", "dask"],
        produces={"x": pa.int32(), "y": pa.int32()},
    )
    class CreateData(DaskLoadComponent):
        def load(self) -> dd.DataFrame:
            df = pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                },
                index=pd.Index(["a", "b", "c"], name="id"),
            )
            return dd.from_pandas(df, npartitions=1)

    dataset = Dataset.create(
        ref=CreateData,
        dataset_name="test-pipeline",
    )

    compiler = KubeFlowCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_spec.yaml")
        compiler.compile(
            dataset=dataset,
            working_directory="/foo/bar",
            output_path=output_path,
        )
        pipeline_configs = KubeflowPipelineConfigs.from_spec(output_path)
        assert pipeline_configs.component_configs["createdata"].image == (
            "python:3.10-slim-buster"
        )
        assert pipeline_configs.component_configs["createdata"].command == [
            "sh",
            "-ec",
            '                printf \'pandas\ndask\' > \'requirements.txt\'\n                python3 -m pip install -r requirements.txt\n            printf \'from typing import *\nimport typing as t\n\nimport dask.dataframe as dd\nimport fondant\nimport pandas as pd\nfrom fondant.component import *\nfrom fondant.core import *\n\n\nclass CreateData(DaskLoadComponent):\n    def load(self) -> dd.DataFrame:\n        df = pd.DataFrame(\n            {\n                "x": [1, 2, 3],\n                "y": [4, 5, 6],\n            },\n            index=pd.Index(["a", "b", "c"], name="id"),\n        )\n        return dd.from_pandas(df, npartitions=1)\n\' > \'main.py\'\n            fondant execute main "$@"\n',  # noqa E501
            "--",
        ]


@pytest.mark.usefixtures("_freeze_time")
def test_kubeflow_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to kubeflow."""
    example_dir, working_directory, dataset, _ = setup_pipeline
    compiler = KubeFlowCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
        )
        pipeline_configs = KubeflowPipelineConfigs.from_spec(output_path)
        assert pipeline_configs.dataset_name == dataset.name
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # Get exepcted component configs
            component = dataset._graph[component_name]
            component_op = component["operation"]

            # Check that the component configs are correct
            assert component_configs.dependencies == component["dependencies"]
            assert component_configs.memory_limit is None
            assert component_configs.memory_request is None
            assert component_configs.cpu_limit is None
            assert component_configs.cpu_request is None
            if component_configs.accelerators:
                assert (
                    component_configs.accelerators.number_of_accelerators
                    == component_op.accelerators.number_of_accelerators
                )
            if component_op.input_partition_rows is not None:
                assert (
                    int(component_configs.arguments["input_partition_rows"])
                    == component_op.input_partition_rows
                )


@pytest.mark.usefixtures("_freeze_time")
def test_kubeflow_configuration(tmp_path_factory):
    """Test that the kubeflow pipeline can be configured."""
    node_pool_label = "dummy_label"
    node_pool_name = "dummy_label"

    dataset = Dataset.create(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            node_pool_label=node_pool_label,
            node_pool_name=node_pool_name,
            accelerator_number=1,
            accelerator_name="GPU",
        ),
        produces={"captions_data": pa.string()},
        dataset_name="test_pipeline",
    )
    compiler = KubeFlowCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(
            dataset=dataset,
            working_directory="/foo/bar",
            output_path=output_path,
        )
        pipeline_configs = KubeflowPipelineConfigs.from_spec(output_path)
        component_configs = pipeline_configs.component_configs["first_component"]
        for accelerator in component_configs.accelerators:
            assert accelerator.type == "nvidia.com/gpu"
            assert accelerator.number == 1
        assert component_configs.node_pool_label == node_pool_label
        assert component_configs.node_pool_name == node_pool_name


@pytest.mark.usefixtures("_freeze_time")
def test_invalid_kubeflow_configuration(tmp_path_factory):
    """Test that an error is returned when an invalid resource is provided."""
    dataset = Dataset.create(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="unknown resource",
        ),
        produces={"captions_data": pa.string()},
        dataset_name="test_pipeline",
    )

    compiler = KubeFlowCompiler()
    with pytest.raises(InvalidDatasetDefinition):
        compiler.compile(
            dataset=dataset,
            working_directory="/foo/bar",
            output_path="kubeflow_pipeline.yml",
        )


def test_kfp_import():
    """Test that the kfp import throws the correct error."""
    with mock.patch.dict(sys.modules):
        # remove kfp from the modules
        sys.modules["kfp"] = None
        with pytest.raises(ImportError):
            _ = KubeFlowCompiler()


@pytest.mark.usefixtures("_freeze_time")
def test_vertex_compiler(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to vertex."""
    example_dir, _, dataset, _ = setup_pipeline
    compiler = VertexCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        working_directory = str(fn)
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
        )
        pipeline_configs = VertexPipelineConfigs.from_spec(output_path)
        assert pipeline_configs.dataset_name == dataset.name
        for (
            component_name,
            component_configs,
        ) in pipeline_configs.component_configs.items():
            # Get exepcted component configs
            component = dataset._graph[component_name]
            component_op = component["operation"]

            # Check that the component configs are correct
            assert component_configs.dependencies == component["dependencies"]
            assert component_configs.memory_limit is None
            assert component_configs.memory_request is None
            assert component_configs.cpu_limit is None
            assert component_configs.cpu_request is None
            if component_configs.accelerators:
                assert (
                    component_configs.accelerators.number_of_accelerators
                    == component_op.accelerators.number_of_accelerators
                )
            if component_op.input_partition_rows is not None:
                assert (
                    int(component_configs.arguments["input_partition_rows"])
                    == component_op.input_partition_rows
                )


@pytest.mark.usefixtures("_freeze_time")
def test_vertex_configuration(tmp_path_factory):
    """Test that the kubeflow pipeline can be configured."""
    dataset = Dataset.create(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="NVIDIA_TESLA_K80",
        ),
        produces={"captions_data": pa.string()},
        dataset_name="test_pipeline",
    )
    compiler = VertexCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        working_directory = str(fn)
        output_path = str(fn / "kubeflow_pipeline.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
        )
        pipeline_configs = VertexPipelineConfigs.from_spec(output_path)
        component_configs = pipeline_configs.component_configs["first_component"]
        for accelerator in component_configs.accelerators:
            assert accelerator.type == "NVIDIA_TESLA_K80"
            assert accelerator.number == "1"


@pytest.mark.usefixtures("_freeze_time")
def test_invalid_vertex_configuration(tmp_path_factory):
    """Test that extra volumes are applied correctly."""
    dataset = Dataset.create(
        Path(COMPONENTS_PATH / "example_1" / "first_component"),
        arguments={"storage_args": "a dummy string arg"},
        resources=Resources(
            accelerator_number=1,
            accelerator_name="unknown resource",
        ),
        produces={"captions_data": pa.string()},
        dataset_name="test_pipeline",
    )
    compiler = VertexCompiler()
    with pytest.raises(InvalidDatasetDefinition):
        compiler.compile(
            dataset=dataset,
            working_directory="/foo/bar",
            output_path="kubeflow_pipeline.yml",
        )


def test_caching_dependency_docker(tmp_path_factory):
    """Test that the component cache key changes when a dependant component cache key change for
    the docker compiler.
    """
    arg_list = ["dummy_arg_1", "dummy_arg_2"]
    second_component_cache_key_dict = {}

    for arg in arg_list:
        compiler = DockerCompiler()

        dataset = Dataset.create(
            Path(COMPONENTS_PATH / "example_1" / "first_component"),
            arguments={"storage_args": f"{arg}"},
            produces={"images_data": pa.binary()},
            dataset_name="test_pipeline",
        )
        dataset.apply(
            Path(COMPONENTS_PATH / "example_1" / "second_component"),
            arguments={"storage_args": "a dummy string arg"},
        )

        with tmp_path_factory.mktemp("temp") as fn:
            working_directory = str(fn)
            output_path = str(fn / "docker-compose.yml")
            compiler.compile(
                dataset=dataset,
                working_directory=working_directory,
                output_path=output_path,
                build_args=[],
            )
            pipeline_configs = DockerComposeConfigs.from_spec(output_path)
            metadata = json.loads(
                pipeline_configs.component_configs["second_component"].arguments[
                    "metadata"
                ],
            )
            cache_key = metadata["cache_key"]
            second_component_cache_key_dict[arg] = cache_key

    assert (
        second_component_cache_key_dict[arg_list[0]]
        != second_component_cache_key_dict[arg_list[1]]
    )


def test_caching_dependency_kfp(tmp_path_factory):
    """Test that the component cache key changes when a depending component cache key change for
    the kubeflow compiler.
    """
    arg_list = ["dummy_arg_1", "dummy_arg_2"]
    second_component_cache_key_dict = {}

    for arg in arg_list:
        compiler = KubeFlowCompiler()

        dataset = Dataset.create(
            Path(COMPONENTS_PATH / "example_1" / "first_component"),
            arguments={"storage_args": f"{arg}"},
            produces={"images_data": pa.binary()},
            dataset_name="test_pipeline",
        )
        dataset.apply(
            Path(COMPONENTS_PATH / "example_1" / "second_component"),
            arguments={"storage_args": "a dummy string arg"},
        )

        with tmp_path_factory.mktemp("temp") as fn:
            output_path = str(fn / "kubeflow_pipeline.yml")
            compiler.compile(
                dataset=dataset,
                working_directory="/foo/bar",
                output_path=output_path,
            )
            pipeline_configs = KubeflowPipelineConfigs.from_spec(output_path)

            metadata = json.loads(
                pipeline_configs.component_configs["second_component"].arguments[
                    "metadata"
                ],
            )
            cache_key = metadata["cache_key"]
            second_component_cache_key_dict[arg] = cache_key
        second_component_cache_key_dict[arg] = cache_key

    assert (
        second_component_cache_key_dict[arg_list[0]]
        != second_component_cache_key_dict[arg_list[1]]
    )


def test_sagemaker_build_command():
    compiler = SagemakerCompiler()
    metadata = Metadata(
        dataset_name="example_pipeline",
        manifest_location="/foo/bar/manifest.json",
        component_id="component_2",
        run_id="example_pipeline_2024",
        cache_key="42",
    )
    args = {"foo": "bar", "baz": "qux"}
    command = compiler._build_command(
        metadata=metadata,
        arguments=args,
        working_directory="/foo/bar",
    )

    assert command == [
        "--metadata",
        '\'{"dataset_name": "example_pipeline", "run_id": "example_pipeline_2024", '
        '"component_id": "component_2", "cache_key": "42", "manifest_location": '
        '"/foo/bar/manifest.json"}\'',
        "--output_manifest_path",
        "/foo/bar/example_pipeline/example_pipeline_2024/component_2/manifest.json",
        "--foo",
        "'bar'",
        "--baz",
        "'qux'",
        "--working_directory",
        "/foo/bar",
    ]
    # with dependencies
    dependencies = ["component_1"]

    command2 = compiler._build_command(
        metadata=metadata,
        arguments=args,
        dependencies=dependencies,
        working_directory="/foo/bar",
    )

    assert command2 == [
        "--metadata",
        '\'{"dataset_name": "example_pipeline", "run_id": "example_pipeline_2024", '
        '"component_id": "component_2", "cache_key": "42", "manifest_location": '
        '"/foo/bar/manifest.json"}\'',
        "--output_manifest_path",
        "/foo/bar/example_pipeline/example_pipeline_2024/component_2/manifest.json",
        "--foo",
        "'bar'",
        "--baz",
        "'qux'",
        "--input_manifest_path",
        "/foo/bar/example_pipeline/example_pipeline_2024/component_1/manifest.json",
        "--working_directory",
        "/foo/bar",
    ]


def test_sagemaker_generate_script(tmp_path_factory):
    compiler = SagemakerCompiler()
    command = ["--metadata", '{"foo": "bar\'s"}']
    with tmp_path_factory.mktemp("temp") as fn:
        script_path = compiler.generate_component_script(
            entrypoint=["fondant", "execute", "main"],
            command=command,
            component_name="component_1",
            directory=fn,
        )

        assert script_path == f"{fn}/component_1.sh"

        assert not subprocess.check_call(["bash", "-n", script_path])  # nosec

        with open(script_path) as f:
            assert (
                f.read()
                == 'fondant execute main --metadata \'{"foo": "bars"}\''  # E501
            )


def test_sagemaker_generate_script_lightweight_component(tmp_path_factory):
    @lightweight_component(
        base_image="python:3.10-slim-buster",
        extra_requires=["pandas", "dask"],
    )
    class CreateData(DaskLoadComponent):
        def load(self) -> dd.DataFrame:
            df = pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                },
                index=pd.Index(["a", "b", "c"], name="id"),
            )
            return dd.from_pandas(df, npartitions=1)

    component_op = ComponentOp.from_ref(
        ref=CreateData,
        produces={"x": pa.int32(), "y": pa.int32()},
    )

    compiler = SagemakerCompiler()

    metadata = Metadata(
        dataset_name="example_pipeline",
        manifest_location="/foo/bar/manifest.json",
        component_id="component_2",
        run_id="example_pipeline_2024",
        cache_key="42",
    )
    args = {}

    with tmp_path_factory.mktemp("temp") as fn:
        script_path = compiler.generate_component_script(
            entrypoint=compiler._build_entrypoint(component_op.image),
            command=compiler._build_command(
                metadata=metadata,
                arguments=args,
                working_directory=str(fn),
            ),
            component_name=component_op.component_name,
            directory=fn,
        )

        assert script_path == f"{fn}/{component_op.component_name}.sh"

        assert not subprocess.check_call(["bash", "-n", script_path])  # nosec


def test_sagemaker_base_path_validator():
    compiler = SagemakerCompiler()

    # no lowercase 's3'
    with pytest.raises(
        ValueError,
        match="base_path must be a valid s3 path, starting with s3://",
    ):
        compiler.validate_base_path("S3://foo/bar")
    # ends with '/'
    with pytest.raises(ValueError, match="base_path must not end with a '/'"):
        compiler.validate_base_path("s3://foo/bar/")

    # valid
    compiler.validate_base_path("s3://foo/bar")


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler_create_local_base_path(setup_pipeline, tmp_path_factory):
    """Test compiling a pipeline to docker-compose."""
    example_dir, workspace, dataset, _ = setup_pipeline
    compiler = DockerCompiler()
    with tmp_path_factory.mktemp("temp") as fn:
        working_directory = str(fn) + "/my-artifacts"
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
            build_args=[],
        )
        assert Path(working_directory).exists()


@pytest.mark.usefixtures("_freeze_time")
def test_docker_compiler_create_local_base_path_propagate_exception(
    setup_pipeline,
    tmp_path_factory,
):
    """Test compiling a pipeline to docker-compose."""
    example_dir, _, dataset, _ = setup_pipeline
    compiler = DockerCompiler()
    msg = re.escape(
        "Unable to create and mount local base path. ",
    )

    with tmp_path_factory.mktemp("temp") as fn, pytest.raises(  # noqa PT012
        ValueError,
        match=msg,
    ):
        working_directory = "/my-artifacts"
        output_path = str(fn / "docker-compose.yml")
        compiler.compile(
            dataset=dataset,
            working_directory=working_directory,
            output_path=output_path,
            build_args=[],
        )
