import argparse
import subprocess
from unittest.mock import patch

import pytest
from fondant.cli import ImportFromStringError, compile, pipeline_from_string, run
from fondant.pipeline import Pipeline

commands = [
    "fondant",
    "fondant --help",
    "fondant explore --help",
    "fondant compile --help",
    "fondant run --help",
]


@pytest.mark.parametrize("command", commands)
def test_basic_invocation(command):
    """Test that the CLI (sub)commands can be invoked without errors."""
    process = subprocess.run(command, shell=True, capture_output=True)
    assert process.returncode == 0


TEST_PIPELINE = Pipeline(pipeline_name="test_pipeline", base_path="some/path")


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


def test_compile_logic(tmp_path_factory):
    """Test that the compile command works with arguments."""
    with tmp_path_factory.mktemp("temp") as fn:
        args = argparse.Namespace(
            local=True,
            pipeline=TEST_PIPELINE,
            output_path=str(fn / "docker-compose.yml"),
            extra_volumes=[],
            run_id=None,
            resume_component=None,
        )
        compile(args)
    args2 = argparse.Namespace(kubeflow=True, local=False, ref="some/path")
    with pytest.raises(NotImplementedError):
        compile(args2)


def test_run_logic(tmp_path_factory):
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
    args2 = argparse.Namespace(kubeflow=True, local=False, ref="some/path")
    with pytest.raises(NotImplementedError):
        run(args2)
