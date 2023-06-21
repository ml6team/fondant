import subprocess

import pytest

commands = [
    "fondant --help",
    "fondant explore --help",
]


@pytest.mark.parametrize("command", commands)
def test_basic_invocation(command):
    """Test that the CLI can be invoked without errors."""
    process = subprocess.run(command, shell=True, capture_output=True)
    assert process.returncode == 0
