from pathlib import Path
from unittest.mock import patch

import pytest
from fondant.core.schema import CloudCredentialsMount
from fondant.explore import run_explorer_app, stop_explorer_app
from fondant.testing import DockerComposeConfigs

DEFAULT_CONTAINER = "fndnt/data_explorer"
OUTPUT_FILE = "explorer-compose.yaml"
DEFAULT_TAG = "latest"
DEFAULT_PORT = 8501


@pytest.fixture()
def host_path() -> str:
    return "tests/path/to/source"


@pytest.fixture()
def remote_path() -> str:
    return "gs://bucket/path/to/source"


@pytest.fixture()
def container_path() -> str:
    return "/source"


def test_run_data_explorer_local_base_path(
    host_path,
    container_path,
    tmp_path_factory,
):
    """Test that the data explorer can be run with a local base path."""
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "subprocess.check_call",
    ) as mock_call:
        output_path = str(fn / OUTPUT_FILE)
        run_explorer_app(
            base_path=host_path,
            output_path=output_path,
            port=DEFAULT_PORT,
            container=DEFAULT_CONTAINER,
            tag=DEFAULT_TAG,
        )

        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        data_explorer_config = pipeline_configs.component_configs["data_explorer"]
        volumes = data_explorer_config.volumes

        assert data_explorer_config.arguments["base_path"] == container_path
        assert data_explorer_config.image == f"{DEFAULT_CONTAINER}:{DEFAULT_TAG}"
        assert data_explorer_config.ports == [f"{DEFAULT_PORT}:8501"]
        assert volumes[0]["source"] == str(Path(host_path).resolve())
        assert volumes[0]["target"] == container_path

        mock_call.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                output_path,
                "up",
                "--build",
                "--pull",
                "always",
                "--remove-orphans",
                "--detach",
            ],
            stdout=-1,
        )


def test_run_data_explorer_remote_base_path(
    remote_path,
    tmp_path_factory,
):
    """Test that the data explorer can be run with a remote base path."""
    for auth_provider in CloudCredentialsMount:
        extra_auth_volume = auth_provider.get_path()

        with tmp_path_factory.mktemp("temp") as fn, patch(
            "subprocess.check_call",
        ) as mock_call:
            output_path = str(fn / OUTPUT_FILE)
            run_explorer_app(
                base_path=remote_path,
                output_path=output_path,
                port=DEFAULT_PORT,
                container=DEFAULT_CONTAINER,
                tag=DEFAULT_TAG,
                auth_provider=auth_provider,
            )

            pipeline_configs = DockerComposeConfigs.from_spec(output_path)
            data_explorer_config = pipeline_configs.component_configs["data_explorer"]
            volumes = data_explorer_config.volumes

            assert data_explorer_config.arguments["base_path"] == remote_path
            assert data_explorer_config.image == f"{DEFAULT_CONTAINER}:{DEFAULT_TAG}"
            assert data_explorer_config.ports == [f"{DEFAULT_PORT}:8501"]
            assert volumes[0] == extra_auth_volume

            mock_call.assert_called_once_with(
                [
                    "docker",
                    "compose",
                    "-f",
                    output_path,
                    "up",
                    "--build",
                    "--pull",
                    "always",
                    "--remove-orphans",
                    "--detach",
                ],
                stdout=-1,
            )


def test_stop_data_explorer(
    remote_path,
    tmp_path_factory,
):
    """Test that the data explorer can be run with a remote base path."""
    with tmp_path_factory.mktemp("temp") as fn, patch(
        "subprocess.check_call",
    ) as mock_call:
        output_path = str(fn / OUTPUT_FILE)
        stop_explorer_app(
            output_path=output_path,
        )

        mock_call.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                output_path,
                "stop",
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
            extra_volumes=None,
        )
