import subprocess
from unittest.mock import patch

import pytest

from fondant.cli import DEFAULT_CONTAINER, DEFAULT_PORT, DEFAULT_TAG, run_data_explorer


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
                    "-v",
                    f"{data_directory}:/artifacts",
                    f"{DEFAULT_CONTAINER}:{DEFAULT_TAG}",
                ],
                stdout=subprocess.PIPE,
            )


def test_run_data_explorer_no_data_directory(monkeypatch, caplog):
    monkeypatch.setattr("sys.argv", ["cli.py"])

    with patch("shutil.which") as mock_which:
        with patch("subprocess.call") as mock_call:
            mock_which.return_value = "/usr/bin/docker"

            run_data_explorer()

            assert (
                "You have not provided a data directory."
                + "To access local files, provide a local data directory"
                + " with the --data-directory flag."
                in caplog.text
            )
            mock_call.assert_called_once_with(
                [
                    "docker",
                    "run",
                    "-p",
                    "8501:8501",
                    "ghcr.io/ml6team/data_explorer:latest",
                ],
                stdout=-1,
            )


def test_run_data_explorer_no_docker(monkeypatch, caplog):
    source_dir = "/path/to/source"
    monkeypatch.setattr("sys.argv", ["cli.py", "--data-directory", source_dir])

    with patch("shutil.which") as mock_which:
        mock_which.return_value = None

        run_data_explorer()

        mock_which.assert_called_once_with("docker")
        assert "Docker runtime not found" in caplog.text


def test_run_data_explorer_with_credentials(monkeypatch):
    data_directory = "/path/to/source"
    credentials = "/path/to/credentials"
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "--data-directory", data_directory, "--credentials", credentials],
    )

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
                    "8501:8501",
                    "-v",
                    "/path/to/credentials:ro",
                    "-v",
                    "/path/to/source:/artifacts",
                    "ghcr.io/ml6team/data_explorer:latest",
                ],
                stdout=-1,
            )


if __name__ == "__main__":
    pytest.main()
