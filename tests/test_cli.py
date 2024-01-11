import argparse
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fondant.cli import (
    ComponentImportError,
    PipelineImportError,
    build,
    compile_kfp,
    compile_local,
    compile_sagemaker,
    compile_vertex,
    component_from_module,
    execute,
    get_module,
    pipeline_from_string,
    run_kfp,
    run_local,
    run_vertex,
)
from fondant.component import DaskLoadComponent
from fondant.component.executor import Executor, ExecutorFactory
from fondant.core.schema import CloudCredentialsMount
from fondant.pipeline import Pipeline

commands = [
    "fondant",
    "fondant --help",
    "fondant explore --help",
    "fondant execute --help",
    "fondant compile --help",
    "fondant run --help",
]


class MyTestComponent(DaskLoadComponent):
    def __init__(self, *args):
        pass

    def load(self):
        pass


@pytest.fixture()
def mock_subprocess_run():
    def _mock_subprocess_run(*args, **kwargs):
        class MockCompletedProcess:
            returncode = 0

        return MockCompletedProcess()

    return _mock_subprocess_run


@pytest.mark.parametrize("command", commands)
def test_basic_invocation(command):
    """Test that the CLI (sub)commands can be invoked without errors."""
    process = subprocess.run(command, shell=True, capture_output=True)
    assert process.returncode == 0


TEST_PIPELINE = Pipeline("test_pipeline", base_path="some/path")


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
        "examples.example_modules.pipeline",
        "examples.example_modules.pipeline:pipeline",
        "examples.example_modules.pipeline:create_pipeline",
        "examples.example_modules.pipeline:create_pipeline_with_args('test_pipeline')",
        "examples.example_modules.pipeline:create_pipeline_with_args(name='test_pipeline')",
    ],
)
def test_pipeline_from_module(module_str):
    """Test that pipeline_from_string works."""
    pipeline = pipeline_from_string(module_str)
    assert pipeline.name == "test_pipeline"


@pytest.mark.parametrize(
    "module_str",
    [
        # module does not contain a pipeline instance
        "examples.example_modules.component",
        # module contains many pipeline instances
        "examples.example_modules.invalid_double_pipeline",
        # Factory expects an argument
        "examples.example_modules.pipeline:create_pipeline_with_args",
        # Factory does not expect an argument
        "examples.example_modules.pipeline:create_pipeline('test_pipeline')",
        # Factory does not expect an argument
        "examples.example_modules.pipeline:create_pipeline(name='test_pipeline')",
        # Invalid argument
        "examples.example_modules.pipeline:create_pipeline(name)",
        # Not a variable or function
        "examples.example_modules.pipeline:[]",
        # Attribute doesn't exist
        "examples.example_modules.pipeline:no_pipeline",
        # Attribute is not a valid python name
        "examples.example_modules.pipeline:pipe;line",
        # Not a Pipeline
        "examples.example_modules.pipeline:number",
    ],
)
def test_pipeline_from_module_error(module_str):
    """Test different error cases for pipeline_from_string."""
    with pytest.raises(PipelineImportError):
        pipeline_from_string(module_str)


def test_factory_error_propagated():
    """Test that an error in the factory method is correctly propagated."""
    with pytest.raises(NotImplementedError):
        pipeline_from_string("examples.example_modules.pipeline:not_implemented")


def test_execute_logic(monkeypatch):
    """Test that the execute command works with arguments."""
    args = argparse.Namespace(ref=__name__)
    monkeypatch.setattr(ExecutorFactory, "get_executor", lambda self: Executor)
    monkeypatch.setattr(Executor, "execute", lambda component_cls: None)
    execute(args)


def test_local_compile(tmp_path_factory):
    """Test that the compile command works with arguments."""
    namespace_creds_kwargs = [
        {"auth_gcp": True, "auth_azure": False, "auth_aws": False},
        {"auth_gcp": False, "auth_azure": True, "auth_aws": False},
        {"auth_gcp": False, "auth_azure": False, "auth_aws": True},
    ]

    for namespace_cred_kwargs in namespace_creds_kwargs:
        with tmp_path_factory.mktemp("temp") as fn, patch(
            "fondant.pipeline.compiler.DockerCompiler.compile",
        ) as mock_compiler:
            args = argparse.Namespace(
                ref=__name__,
                local=True,
                kubeflow=False,
                vertex=False,
                output_path=str(fn / "docker-compose.yml"),
                extra_volumes=[],
                build_arg=[],
                **namespace_cred_kwargs,
                credentials=None,
            )
            compile_local(args)

            if namespace_cred_kwargs["auth_gcp"] is True:
                extra_volumes = [CloudCredentialsMount.GCP.value]
            if namespace_cred_kwargs["auth_aws"] is True:
                extra_volumes = [CloudCredentialsMount.AWS.value]
            if namespace_cred_kwargs["auth_azure"] is True:
                extra_volumes = [CloudCredentialsMount.AZURE.value]

            mock_compiler.assert_called_once_with(
                pipeline=TEST_PIPELINE,
                extra_volumes=extra_volumes,
                output_path=str(fn / "docker-compose.yml"),
                build_args=[],
            )


def test_kfp_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.pipeline.compiler.KubeFlowCompiler.compile",
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
            pipeline=TEST_PIPELINE,
            output_path=str(fn / "kubeflow_pipeline.yml"),
        )


def test_vertex_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.pipeline.compiler.VertexCompiler.compile",
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
            pipeline=TEST_PIPELINE,
            output_path=str(fn / "vertex_pipeline.yml"),
        )


def test_sagemaker_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.pipeline.compiler.SagemakerCompiler.compile",
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
            pipeline=TEST_PIPELINE,
            output_path=str(fn / "sagemaker_pipeline.json"),
            role_arn="some_role",
        )


def test_local_run(mock_subprocess_run):
    """Test that the run command works with different arguments."""
    args = argparse.Namespace(
        local=True,
        ref="some/path",
        output_path=None,
        auth_gcp=False,
        auth_azure=False,
        auth_aws=False,
        credentials=None,
        extra_volumes=[],
        build_arg=[],
    )

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = mock_subprocess_run
        run_local(args)
        mock_run.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                "some/path",
                "up",
                "--build",
                "--pull",
                "always",
                "--remove-orphans",
                "--abort-on-container-exit",
            ],
            env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
            encoding="utf8",
        )

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = mock_subprocess_run
        args1 = argparse.Namespace(
            local=True,
            vertex=False,
            kubeflow=False,
            ref=__name__,
            extra_volumes=[],
            build_arg=[],
            auth_gcp=False,
            auth_azure=False,
            auth_aws=False,
            credentials=None,
        )
        run_local(args1)
        mock_run.assert_called_once_with(
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
                "--abort-on-container-exit",
            ],
            env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
            encoding="utf8",
        )


def test_local_run_cloud_credentials(mock_subprocess_run):
    namespace_creds_kwargs = [
        {"auth_gcp": True, "auth_azure": False, "auth_aws": False},
        {"auth_gcp": False, "auth_azure": True, "auth_aws": False},
        {"auth_gcp": False, "auth_azure": False, "auth_aws": True},
    ]

    for namespace_cred_kwargs in namespace_creds_kwargs:
        with patch(
            "fondant.pipeline.compiler.DockerCompiler.compile",
        ) as mock_compiler, patch(
            "subprocess.run",
        ) as mock_runner:
            mock_runner.side_effect = mock_subprocess_run

            args = argparse.Namespace(
                local=True,
                vertex=False,
                kubeflow=False,
                ref=__name__,
                **namespace_cred_kwargs,
                credentials=None,
                extra_volumes=[],
                build_arg=[],
            )
            run_local(args)

            if namespace_cred_kwargs["auth_gcp"] is True:
                extra_volumes = [CloudCredentialsMount.GCP.value]
            if namespace_cred_kwargs["auth_aws"] is True:
                extra_volumes = [CloudCredentialsMount.AWS.value]
            if namespace_cred_kwargs["auth_azure"] is True:
                extra_volumes = [CloudCredentialsMount.AZURE.value]

            mock_compiler.assert_called_once_with(
                TEST_PIPELINE,
                extra_volumes=extra_volumes,
                output_path=".fondant/compose.yaml",
                build_args=[],
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
                    "--abort-on-container-exit",
                ],
                env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
                encoding="utf8",
            )


def test_kfp_run(tmp_path_factory):
    """Test that the run command works in different scenarios."""
    args = argparse.Namespace(
        kubeflow=True,
        local=False,
        vertex=False,
        output_path=None,
        ref="some/path",
        host=None,
    )
    with pytest.raises(
        ValueError,
        match="--host argument is required for running on Kubeflow",
    ):  # no host
        run_kfp(args)
    with patch("fondant.pipeline.runner.KubeflowRunner") as mock_runner:
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            output_path=None,
            host="localhost",
            ref="some/path",
        )
        run_kfp(args)
        mock_runner.assert_called_once_with(host="localhost")
    with patch("fondant.pipeline.runner.KubeflowRunner") as mock_runner, patch(
        "fondant.pipeline.compiler.KubeFlowCompiler",
    ) as mock_compiler, tmp_path_factory.mktemp(
        "temp",
    ) as fn:
        mock_compiler.compile.return_value = "some/path"
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            host="localhost2",
            output_path=str(fn / "kubeflow_pipelines.yml"),
            ref=__name__,
        )
        run_kfp(args)
        mock_runner.assert_called_once_with(host="localhost2")


def test_vertex_run(tmp_path_factory):
    """Test that the run command works in different scenarios."""
    with patch("fondant.pipeline.runner.VertexRunner") as mock_runner:
        args = argparse.Namespace(
            kubeflow=False,
            local=False,
            vertex=True,
            output_path=None,
            region="europe-west-1",
            project_id="project-123",
            service_account=None,
            network=None,
            ref="some/path",
        )
        run_vertex(args)
        mock_runner.assert_called_once_with(
            project_id="project-123",
            region="europe-west-1",
            service_account=None,
            network=None,
        )

    with patch("fondant.pipeline.runner.VertexRunner") as mock_runner, patch(
        "fondant.pipeline.compiler.VertexCompiler",
    ) as mock_compiler, tmp_path_factory.mktemp(
        "temp",
    ) as fn:
        mock_compiler.compile.return_value = "some/path"
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            host="localhost2",
            output_path=str(fn / "kubeflow_pipelines.yml"),
            ref=__name__,
            region="europe-west-1",
            project_id="project-123",
            service_account=None,
            network=None,
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
