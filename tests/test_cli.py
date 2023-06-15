import subprocess
from unittest.mock import patch

import pytest

from fondant.cli import DEFAULT_PORT, DEFAULT_REGISTRY, DEFAULT_TAG, run_data_explorer


def test_run_data_explorer(monkeypatch):
    data_directory = "/path/to/source"
    monkeypatch.setattr("sys.argv", ["cli.py", "--data-directory", data_directory])

    with patch("shutil.which") as mock_which:
        mock_which.return_value = "/usr/bin/docker"

        with patch("subprocess.call") as mock_call:
            run_data_explorer()

            mock_which.assert_called_once_with("docker")
            mock_call.assert_called_once_with(
                [
                    "docker",
                    "run",
                    "-p",
                    f"{DEFAULT_PORT}:8501",
                    "--mount",
                    f"type=bind,source={data_directory},target=/artifacts",
                    f"{DEFAULT_REGISTRY}:{DEFAULT_TAG}",
                ],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )


def test_run_data_explorer_no_source(monkeypatch, caplog):
    monkeypatch.setattr("sys.argv", ["cli.py"])

    with patch("shutil.which") as mock_which:
        mock_which.return_value = "/usr/bin/docker"

        run_data_explorer()

        # mock_which.assert_not_called()
        assert (
            "Please provide a source directory with the --data-directory or -d option."
            in caplog.text
        )


def test_run_data_explorer_no_docker(monkeypatch, caplog):
    source_dir = "/path/to/source"
    monkeypatch.setattr("sys.argv", ["cli.py", "--data-directory", source_dir])

    with patch("shutil.which") as mock_which:
        mock_which.return_value = None

        run_data_explorer()

        mock_which.assert_called_once_with("docker")
        assert "Docker runtime not found" in caplog.text


if __name__ == "__main__":
    pytest.main()
