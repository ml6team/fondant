import argparse
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fondant.cli import (
    ComponentImportError,
    DatasetImportError,
    build,
    compile_kfp,
    compile_local,
    compile_sagemaker,
    compile_vertex,
    component_from_module,
    dataset_from_string,
    execute,
    get_module,
    run_kfp,
    run_local,
    run_vertex,
)
from fondant.component import DaskLoadComponent
from fondant.component.executor import Executor, ExecutorFactory
from fondant.core.manifest import Manifest
from fondant.core.schema import CloudCredentialsMount
from fondant.dataset import Dataset
from fondant.dataset.runner import DockerRunner

commands = [
    "fondant",
    "fondant --help",
    "fondant explore --help",
    "fondant execute --help",
    "fondant compile --help",
    "fondant run --help",
]


@pytest.fixture()
def mock_docker_installation(monkeypatch):  # noqa: PT004
    def mock_check_docker_install(self):
        pass

    def mock_check_docker_compose_install(self):
        pass

    monkeypatch.setattr(DockerRunner, "check_docker_install", mock_check_docker_install)
    monkeypatch.setattr(
        DockerRunner,
        "check_docker_compose_install",
        mock_check_docker_compose_install,
    )


class MyTestComponent(DaskLoadComponent):
    def __init__(self, *args):
        pass

    def load(self):
        pass


@pytest.mark.parametrize("command", commands)
def test_basic_invocation(command):
    """Test that the CLI (sub)commands can be invoked without errors."""
    process = subprocess.run(command, shell=True, capture_output=True)
    assert process.returncode == 0


TEST_MANIFEST = Manifest.create(dataset_name="test_dataset", run_id="test_run_id")
TEST_DATASET = Dataset(manifest=TEST_MANIFEST)


@pytest.mark.parametrize(
    "module_str",
    [
        "examples.example_modules.component",
        "examples.example_modules/component",
        "examples.example_modules.component.py",
        "examples.example_modules/component.py",
    ],
)
def test_get_module(module_str):
    """Test get module method."""
    module = get_module(module_str)
    assert module.__name__ == "examples.example_modules.component"


def test_get_module_error():
    """Test that an error is returned when an attempting to import an invalid module."""
    with pytest.raises(ModuleNotFoundError):
        component_from_module("example_modules.invalid")


@pytest.mark.parametrize(
    "module_str",
    [
        __name__,  # cannot be split
        "examples.example_modules.component",  # module does not exist
    ],
)
def test_component_from_module(module_str):
    """Test that the component from module works."""
    component = component_from_module(module_str)
    assert component.__name__ == "MyTestComponent"


@pytest.mark.parametrize(
    "module_str",
    [
        # module contains more than one component class
        "examples.example_modules.invalid_component",
        # module does not contain a component class
        "examples.example_modules.invalid_double_components",
    ],
)
def test_component_from_module_error(module_str):
    """Test different error cases for pipeline_from_string."""
    with pytest.raises(ComponentImportError):
        component_from_module(module_str)


@pytest.mark.parametrize(
    "module_str",
    [
        __name__,
        "examples.example_modules.dataset",
        "examples.example_modules.dataset:workspace",
        "examples.example_modules.dataset:create_dataset",
        "examples.example_modules.dataset:create_dataset_with_args('test_dataset')",
        "examples.example_modules.dataset:create_dataset_with_args(name='test_dataset')",
    ],
)
def test_pipeline_from_module(module_str):
    """Test that pipeline_from_string works."""
    dataset = dataset_from_string(module_str)
    assert dataset.name == "test_dataset"


@pytest.mark.parametrize(
    "module_str",
    [
        # module does not contain a pipeline instance
        "examples.example_modules.component",
        # Factory expects an argument
        "examples.example_modules.dataset:create_pipeline_with_args",
        # Factory does not expect an argument
        "examples.example_modules.dataset:create_pipeline('test_pipeline')",
        # Factory does not expect an argument
        "examples.example_modules.dataset:create_pipeline(name='test_pipeline')",
        # Invalid argument
        "examples.example_modules.dataset:create_pipeline(name)",
        # Not a variable or function
        "examples.example_modules.dataset:[]",
        # Attribute doesn't exist
        "examples.example_modules.dataset:no_pipeline",
        # Attribute is not a valid python name
        "examples.example_modules.dataset:pipe;line",
        # Not a Pipeline
        "examples.example_modules.dataset:number",
    ],
)
def test_pipeline_from_module_error(module_str):
    """Test different error cases for pipeline_from_string."""
    with pytest.raises(DatasetImportError):
        dataset_from_string(module_str)


def test_factory_error_propagated():
    """Test that an error in the factory method is correctly propagated."""
    with pytest.raises(NotImplementedError):
        dataset_from_string("examples.example_modules.dataset:not_implemented")


def test_execute_logic(monkeypatch):
    """Test that the execute command works with arguments."""
    args = argparse.Namespace(ref=__name__)
    monkeypatch.setattr(ExecutorFactory, "get_executor", lambda self: Executor)
    monkeypatch.setattr(Executor, "execute", lambda component_cls: None)
    execute(args)


def test_local_compile(tmp_path_factory):
    """Test that the compile command works with arguments."""
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.dataset.compiler.DockerCompiler.compile",
    ) as mock_compiler:
        args = argparse.Namespace(
            ref=__name__,
            local=True,
            kubeflow=False,
            vertex=False,
            output_path=str(fn / "docker-compose.yml"),
            extra_volumes=[],
            build_arg=[],
            credentials=None,
            auth_provider=None,
        )
        compile_local(args)

        mock_compiler.assert_called_once_with(
            dataset=TEST_DATASET,
            extra_volumes=[],
            output_path=str(fn / "docker-compose.yml"),
            build_args=[],
            auth_provider=None,
        )


def test_kfp_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.dataset.compiler.KubeFlowCompiler.compile",
    ) as mock_compiler:
        args = argparse.Namespace(
            ref=__name__,
            kubeflow=True,
            local=False,
            vertex=False,
            output_path=str(fn / "kubeflow_pipeline.yml"),
        )
        compile_kfp(args)
        mock_compiler.assert_called_once_with(
            dataset=TEST_DATASET,
            output_path=str(fn / "kubeflow_pipeline.yml"),
        )


def test_vertex_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.dataset.compiler.VertexCompiler.compile",
    ) as mock_compiler:
        args = argparse.Namespace(
            ref=__name__,
            kubeflow=False,
            local=False,
            vertex=True,
            output_path=str(fn / "vertex_pipeline.yml"),
        )
        compile_vertex(args)
        mock_compiler.assert_called_once_with(
            dataset=TEST_DATASET,
            output_path=str(fn / "vertex_pipeline.yml"),
        )


def test_sagemaker_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.dataset.compiler.SagemakerCompiler.compile",
    ) as mock_compiler:
        args = argparse.Namespace(
            ref=__name__,
            kubeflow=False,
            local=False,
            vertex=False,
            sagemaker=True,
            output_path=str(fn / "sagemaker_pipeline.json"),
            role_arn="some_role",
        )
        compile_sagemaker(args)
        mock_compiler.assert_called_once_with(
            dataset=TEST_DATASET,
            output_path=str(fn / "sagemaker_pipeline.json"),
            role_arn="some_role",
        )


def test_local_run(mock_docker_installation):
    """Test that the run command works with different arguments."""
    args = argparse.Namespace(
        local=True,
        ref=__name__,
        output_path=None,
        auth_provider=None,
        credentials=None,
        extra_volumes=[],
        build_arg=[],
        working_directory="./dummy-dir",
    )
    with patch("subprocess.call") as mock_call:
        run_local(args)
        mock_call.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                ".fondant/compose.yaml",
                "up",
                "--build",
                "--pull",
                "always",
                "--remove-orphans",
            ],
            env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
        )

    with patch("subprocess.call") as mock_call:
        args1 = argparse.Namespace(
            local=True,
            vertex=False,
            kubeflow=False,
            ref=__name__,
            extra_volumes=[],
            build_arg=[],
            auth_provider=None,
            credentials=None,
            working_directory="./dummy-dir",
        )
        run_local(args1)
        mock_call.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                ".fondant/compose.yaml",
                "up",
                "--build",
                "--pull",
                "always",
                "--remove-orphans",
            ],
            env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
        )


def test_local_run_cloud_credentials(mock_docker_installation):
    for auth_provider in CloudCredentialsMount:
        with patch(
            "fondant.dataset.compiler.DockerCompiler.compile",
        ) as mock_compiler, patch(
            "subprocess.call",
        ) as mock_runner:
            args = argparse.Namespace(
                local=True,
                vertex=False,
                kubeflow=False,
                ref=__name__,
                auth_provider=auth_provider,
                credentials=None,
                extra_volumes=[],
                build_arg=[],
                working_directory="dummy-dir",
            )
            run_local(args)

            mock_compiler.assert_called_once_with(
                TEST_DATASET,
                working_directory="dummy-dir",
                output_path=".fondant/compose.yaml",
                extra_volumes=[],
                build_args=[],
                auth_provider=auth_provider,
            )

            mock_runner.assert_called_once_with(
                [
                    "docker",
                    "compose",
                    "-f",
                    ".fondant/compose.yaml",
                    "up",
                    "--build",
                    "--pull",
                    "always",
                    "--remove-orphans",
                ],
                env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
            )


def test_kfp_run(tmp_path_factory):
    """Test that the run command works in different scenarios."""
    args = argparse.Namespace(
        kubeflow=True,
        local=False,
        vertex=False,
        output_path=None,
        ref="dataset",
        working_directory="./dummy-dir",
        host=None,
    )
    with pytest.raises(
        ValueError,
        match="--host argument is required for running on Kubeflow",
    ):  # no host
        run_kfp(args)
    with patch("fondant.dataset.runner.KubeflowRunner") as mock_runner:
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            output_path=None,
            host="localhost",
            ref="dataset",
            working_directory="./dummy-dir",
        )
        run_kfp(args)
        mock_runner.assert_called_once_with(host="localhost")
    with patch("fondant.dataset.runner.KubeflowRunner") as mock_runner, patch(
        "fondant.dataset.compiler.KubeFlowCompiler",
    ) as mock_compiler, tmp_path_factory.mktemp(
        "temp",
    ) as fn:
        mock_compiler.compile.return_value = "some/path"
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            host="localhost2",
            output_path=str(fn / "kubeflow_pipelines.yml"),
            ref="dataset",
            working_directory="./dummy-dir",
        )
        run_kfp(args)
        mock_runner.assert_called_once_with(host="localhost2")


def test_vertex_run(tmp_path_factory):
    """Test that the run command works in different scenarios."""
    with patch("fondant.dataset.runner.VertexRunner") as mock_runner:
        args = argparse.Namespace(
            kubeflow=False,
            local=False,
            vertex=True,
            output_path=None,
            region="europe-west-1",
            project_id="project-123",
            service_account=None,
            network=None,
            ref="dataset",
            working_directory="./dummy-dir",
        )
        run_vertex(args)
        mock_runner.assert_called_once_with(
            project_id="project-123",
            region="europe-west-1",
            service_account=None,
            network=None,
        )

    with patch("fondant.dataset.runner.VertexRunner") as mock_runner, patch(
        "fondant.dataset.compiler.VertexCompiler",
    ) as mock_compiler, tmp_path_factory.mktemp(
        "temp",
    ) as fn:
        mock_compiler.compile.return_value = "some/path"
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            host="localhost2",
            output_path=str(fn / "kubeflow_pipelines.yml"),
            ref="dataset",
            region="europe-west-1",
            project_id="project-123",
            service_account=None,
            network=None,
            working_directory="./dummy-dir",
        )
        run_vertex(args)
        mock_runner.assert_called_once_with(
            project_id="project-123",
            region="europe-west-1",
            service_account=None,
            network=None,
        )


@patch("docker.api.client.APIClient.push")
@patch("docker.api.client.APIClient.build")
def test_component_build(mock_build, mock_push):
    """Test that the build command works as expected."""
    args = argparse.Namespace(
        component_dir=Path(__file__).parent / "examples/example_component",
        tag="image:test",
        build_arg=["key=value"],
        nocache=True,
        pull=True,
        target="base",
        label=["label_0_key=label_0_value", "label_1_key=label_1_value"],
    )

    # Set up the return values for the mocked methods
    mock_build.return_value = ["Dummy logs build"]
    mock_push.return_value = [{"status": "dummy log status"}]

    # Run build command
    build(args)

    # Check that docker build and push were executed correctly
    mock_build.assert_called_with(
        path=str(Path(__file__).parent / "examples/example_component"),
        tag="image:test",
        buildargs={"key": "value"},
        nocache=True,
        pull=True,
        target="base",
        decode=True,
        labels={"label_0_key": "label_0_value", "label_1_key": "label_1_value"},
    )

    mock_push.assert_called_with("image", tag="test", stream=True, decode=True)

    # Check that the component specification file was updated correctly
    with open(
        Path(__file__).parent / "examples/example_component" / "fondant_component.yaml",
        "r+",
    ) as f:
        content = f.read()
        assert "image:test" in content

        # Revert image name in component specification
        content = content.replace("image:test", "image:local")
        f.seek(0)
        f.write(content)
        f.truncate()
