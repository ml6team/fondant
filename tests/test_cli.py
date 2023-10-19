import argparse
import subprocess
from unittest.mock import patch

import pytest
from fondant.cli import (
    ComponentImportError,
    PipelineImportError,
    compile_kfp,
    compile_local,
    compile_vertex,
    component_from_module,
    execute,
    get_module,
    pipeline_from_module,
    run_kfp,
    run_local,
    run_vertex,
)
from fondant.component import DaskLoadComponent
from fondant.executor import Executor, ExecutorFactory
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


@pytest.mark.parametrize("command", commands)
def test_basic_invocation(command):
    """Test that the CLI (sub)commands can be invoked without errors."""
    process = subprocess.run(command, shell=True, capture_output=True)
    assert process.returncode == 0


TEST_PIPELINE = Pipeline(pipeline_name="test_pipeline", base_path="some/path")


@pytest.mark.parametrize(
    "module_str",
    [
        "example_modules.component",
        "example_modules/component",
        "example_modules.component.py",
        "example_modules/component.py",
    ],
)
def test_get_module(module_str):
    """Test get module method."""
    module = get_module(module_str)
    assert module.__name__ == "example_modules.component"


def test_get_module_error():
    """Test that an error is returned when an attempting to import an invalid module."""
    with pytest.raises(ModuleNotFoundError):
        component_from_module("example_modules.invalid")


@pytest.mark.parametrize(
    "module_str",
    [
        __name__,  # cannot be split
        "example_modules.component",  # module does not exist
    ],
)
def test_component_from_module(module_str):
    """Test that the component from module works."""
    component = component_from_module(module_str)
    assert component.__name__ == "MyTestComponent"


@pytest.mark.parametrize(
    "module_str",
    [
        "example_modules.invalid_component",  # module contains more than one component class
        "example_modules.invalid_double_components",  # module does not contain a component class
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
        "example_modules.pipeline",
    ],
)
def test_pipeline_from_module(module_str):
    """Test that pipeline_from_string works."""
    pipeline = pipeline_from_module(module_str)
    assert pipeline.name == "test_pipeline"


@pytest.mark.parametrize(
    "module_str",
    [
        "example_modules.component",  # module does not contain a pipeline instance
        "example_modules.invalid_double_pipeline",  # module contains many pipeline instances
    ],
)
def test_pipeline_from_module_error(module_str):
    """Test different error cases for pipeline_from_string."""
    with pytest.raises(PipelineImportError):
        pipeline_from_module(module_str)


def test_execute_logic(monkeypatch):
    """Test that the execute command works with arguments."""
    args = argparse.Namespace(ref=__name__)
    monkeypatch.setattr(ExecutorFactory, "get_executor", lambda self: Executor)
    monkeypatch.setattr(Executor, "execute", lambda component_cls: None)
    execute(args)


def test_local_logic(tmp_path_factory):
    """Test that the compile command works with arguments."""
    with tmp_path_factory.mktemp("temp") as fn:
        args = argparse.Namespace(
            ref=__name__,
            local=True,
            kubeflow=False,
            vertex=False,
            output_path=str(fn / "docker-compose.yml"),
            extra_volumes=[],
            build_arg=[],
        )
        compile_local(args)


def test_kfp_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "fondant.compiler.KubeFlowCompiler.compile",
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
        "fondant.compiler.VertexCompiler.compile",
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


def test_local_run(tmp_path_factory):
    """Test that the run command works with different arguments."""
    args = argparse.Namespace(local=True, ref="some/path", output_path=None)
    with patch("subprocess.call") as mock_call:
        run_local(args)
        mock_call.assert_called_once_with(
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
            ],
        )

    with patch("subprocess.call") as mock_call, tmp_path_factory.mktemp("temp") as fn:
        args1 = argparse.Namespace(
            local=True,
            vertex=False,
            kubeflow=False,
            ref=__name__,
            output_path=str(fn / "docker-compose.yml"),
            extra_volumes=[],
            build_arg=[],
        )
        run_local(args1)
        mock_call.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                str(fn / "docker-compose.yml"),
                "up",
                "--build",
                "--pull",
                "always",
                "--remove-orphans",
            ],
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
    with patch("fondant.cli.KubeflowRunner") as mock_runner:
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            output_path=None,
            host="localhost",
            ref="some/path",
        )
        run_kfp(args)
        mock_runner.assert_called_once_with(host="localhost")
    with patch("fondant.cli.KubeflowRunner") as mock_runner, patch(
        "fondant.cli.KubeFlowCompiler",
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
    with patch("fondant.cli.VertexRunner") as mock_runner:
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

    with patch("fondant.cli.VertexRunner") as mock_runner, patch(
        "fondant.cli.VertexCompiler",
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
