import argparse
import subprocess
from unittest.mock import patch

import pytest
from fondant.cli import (
    ImportFromModuleError,
    ImportFromStringError,
    compile,
    component_from_module,
    execute,
    pipeline_from_string,
    run,
)
from fondant.component import DaskLoadComponent
from fondant.pipeline import Pipeline
from fondant.runner import ComponentRunner

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
        __name__,
        "example_modules.valid",
        "example_modules/valid",
        "example_modules.valid.py",
        "example_modules/valid.py",
    ],
)
def test_component_from_module(module_str):
    """Test that the component from module works."""
    component = component_from_module(module_str)
    assert component.__name__ == "MyTestComponent"


@pytest.mark.parametrize(
    "module_str",
    [
        "example_modules.None_existing_module",  # module does not exist
        "example_modules.invalid_component",  # module contains more than one component class
        "example_modules.invalid_double_components",  # module does not contain a component class
    ],
)
def test_component_from_module_error(module_str):
    """Test different error cases for pipeline_from_string."""
    with pytest.raises(ImportFromModuleError):
        component_from_module(module_str)


def test_pipeline_from_string():
    """Test that pipeline_from_string works."""
    pipeline = pipeline_from_string(__name__ + ":TEST_PIPELINE")
    assert pipeline == TEST_PIPELINE


@pytest.mark.parametrize(
    "import_string",
    [
        "foo.barTEST_PIPELINE",  # cannot be split
        "foo.bar:TEST_PIPELINE",  # module does not exist
        __name__ + ":IM_NOT_REAL",  # pipeline does not exist
        __name__ + ":test_basic_invocation",  # not a pipeline instance
    ],
)
def test_pipeline_from_string_error(import_string):
    """Test different error cases for pipeline_from_string."""
    with pytest.raises(ImportFromStringError):
        pipeline_from_string(import_string)


def test_execute_logic(monkeypatch):
    """Test that the execute command works with arguments."""
    args = argparse.Namespace(ref=__name__)
    monkeypatch.setattr(ComponentRunner, "run", lambda self: None)
    execute(args)


def test_local_logic(tmp_path_factory):
    """Test that the compile command works with arguments."""
    with tmp_path_factory.mktemp("temp") as fn:
        args = argparse.Namespace(
            local=True,
            kubeflow=False,
            pipeline=TEST_PIPELINE,
            output_path=str(fn / "docker-compose.yml"),
            extra_volumes=[],
            build_arg=[],
        )
        compile(args)


def test_kfp_compile(tmp_path_factory):
    with tmp_path_factory.mktemp("temp") as fn:
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            pipeline=TEST_PIPELINE,
            output_path=str(fn / "kubeflow_pipelines.yml"),
        )
        compile(args)


def test_local_run(tmp_path_factory):
    """Test that the run command works with different arguments."""
    args = argparse.Namespace(local=True, ref="some/path")
    with patch("subprocess.call") as mock_call:
        run(args)
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
            ref=__name__ + ":TEST_PIPELINE",
            output_path=str(fn / "docker-compose.yml"),
            extra_volumes=[],
            build_arg=[],
        )
        run(args1)
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
        ref="some/path",
        host=None,
    )
    with pytest.raises(
        ValueError,
        match="--host argument is required for running on Kubeflow",
    ):  # no host
        run(args)
    with patch("fondant.cli.KubeflowRunner") as mock_runner:
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            host="localhost",
            ref="some/path",
        )
        run(args)
        mock_runner.assert_called_once_with(host="localhost")
    with patch("fondant.cli.KubeflowRunner") as mock_runner, tmp_path_factory.mktemp(
        "temp",
    ) as fn:
        args = argparse.Namespace(
            kubeflow=True,
            local=False,
            host="localhost2",
            output_path=str(fn / "kubeflow_pipelines.yml"),
            ref=__name__ + ":TEST_PIPELINE",
        )
        run(args)
        mock_runner.assert_called_once_with(host="localhost2")
