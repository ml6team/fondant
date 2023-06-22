import subprocess

import pytest

from fondant.cli import ImportFromStringError, pipeline_from_string
from fondant.pipeline import Pipeline

commands = [
    "fondant --help",
    "fondant explore --help",
    "fondant compile --help",
]


@pytest.mark.parametrize("command", commands)
def test_basic_invocation(command):
    """Test that the CLI (sub)commands can be invoked without errors."""
    process = subprocess.run(command, shell=True, capture_output=True)
    assert process.returncode == 0


TEST_PIPELINE = Pipeline(pipeline_name="test_pipeline", base_path="some/path")


def test_pipeline_from_string():
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
    with pytest.raises(ImportFromStringError):
        pipeline_from_string(import_string)
