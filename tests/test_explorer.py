from pathlib import Path
from unittest.mock import patch

import pytest
from fondant.explorer import run_explorer_app

DEFAULT_CONTAINER = "fndnt/data_explorer"
DEFAULT_TAG = "latest"
DEFAULT_PORT = 8501


@pytest.fixture()
def host_path() -> str:
    return "tests/path/to/source"


@pytest.fixture()
def remote_path() -> str:
    return "gs://bucket/path/to/source"


@pytest.fixture()
def credentials() -> str:
    return (
        "$HOME/.config/gcloud/application_default_credentials.json:/root/.config/"
        "gcloud/application_default_credentials.json"
    )


@pytest.fixture()
def container_path() -> str:
    return "/source"


def test_run_data_explorer_local_base_path(host_path, container_path, credentials):
    """Test that the data explorer can be run with a local base path."""
    with patch("subprocess.call") as mock_call:
        run_explorer_app(
            base_path=host_path,
            port=DEFAULT_PORT,
            container=DEFAULT_CONTAINER,
            tag=DEFAULT_TAG,
            credentials=credentials,
        )
        mock_call.assert_called_once_with(
            [
                "docker",
                "run",
                "--pull",
                "always",
                "--name",
                "fondant-explorer",
                "--rm",
                "-p",
                "8501:8501",
                "-v",
                f"{credentials}:ro",
                "-v",
                f"{Path(host_path).resolve()}:{container_path}",
                f"{DEFAULT_CONTAINER}:{DEFAULT_TAG}",
                "--base_path",
                f"{container_path}",
            ],
            stdout=-1,
        )


def test_run_data_explorer_remote_base_path(remote_path, credentials):
    """Test that the data explorer can be run with a remote base path."""
    with patch("subprocess.call") as mock_call:
        run_explorer_app(
            base_path=remote_path,
            port=DEFAULT_PORT,
            container=DEFAULT_CONTAINER,
            tag=DEFAULT_TAG,
            credentials=credentials,
        )

        mock_call.assert_called_once_with(
            [
                "docker",
                "run",
                "--pull",
                "always",
                "--name",
                "fondant-explorer",
                "--rm",
                "-p",
                "8501:8501",
                "-v",
                f"{credentials}:ro",
                f"{DEFAULT_CONTAINER}:{DEFAULT_TAG}",
                "--base_path",
                f"{remote_path}",
            ],
            stdout=-1,
        )


def test_invalid_run_data_explorer_remote_base_path(remote_path):
    """Test that an error is returned when attempting to use the remote base with no mounted
    credentials.
    """
    expected_msg = (
        f"Specified base path `{remote_path}`, Please provide valid credentials when"
        f" using a remote base path"
    )
    with pytest.raises(RuntimeError, match=expected_msg):
        run_explorer_app(
            base_path=remote_path,
            port=DEFAULT_PORT,
            container=DEFAULT_CONTAINER,
            tag=DEFAULT_TAG,
            credentials=None,
        )
