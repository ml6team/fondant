import logging
import os
import shlex
import subprocess  # nosec
import typing as t
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import fsspec.core
import yaml
from fsspec.implementations.local import LocalFileSystem

from fondant.core.schema import CloudCredentialsMount, DockerVolume

CONTAINER = "fndnt/data_explorer"
PORT = 8501
OUTPUT_PATH = ".fondant/explorer-compose.yaml"


def _get_service_name(container: str) -> str:
    return os.path.basename(container)


def _generate_explorer_spec(
    *,
    base_path: str,
    port: int = PORT,
    container: str = CONTAINER,
    tag: t.Optional[str] = None,
    extra_volumes: t.Union[t.Optional[list], t.Optional[str]] = None,
    auth_provider: t.Optional[CloudCredentialsMount] = None,
) -> t.Dict[str, t.Any]:
    """Generate a Docker Compose specification for the Explorer App."""
    if tag is None:
        tag = version("fondant") if version("fondant") != "0.1.dev0" else "latest"

    if extra_volumes is None:
        extra_volumes = []

    if isinstance(extra_volumes, str):
        extra_volumes = [extra_volumes]

    if auth_provider:
        extra_volumes.append(auth_provider.get_path())

    # Mount extra volumes to the container
    volumes: t.List[t.Union[str, dict]] = []

    fs, _ = fsspec.core.url_to_fs(base_path)
    if isinstance(fs, LocalFileSystem):
        logging.info(f"Using local base path: {base_path}")
        logging.info(
            "This directory will be mounted to /artifacts in the container.",
        )
        data_directory_path = Path(base_path).resolve()
        container_path = os.path.join("/", data_directory_path.name)

        command = ["--base_path", f"{container_path}"]

        if extra_volumes:
            volumes.extend(extra_volumes)

        # Mount the local base path to the container
        volume = DockerVolume(
            type="bind",
            source=str(data_directory_path),
            target=f"/{data_directory_path.stem}",
        )
        volumes.append(asdict(volume))

    else:
        if not extra_volumes:
            raise RuntimeError(
                None,
                f"Specified base path `{base_path}`, Please provide valid credentials when using"
                f" a remote base path",
            )

        logging.info(f"Using remote base path: {base_path}")

        # Mount extra volumes to the container
        volumes.extend(extra_volumes)

        # Add the remote base path as a container argument
        command = ["--base_path", base_path]

    services = {
        f"{_get_service_name(container)}": {
            "command": command,
            "volumes": volumes,
            "ports": [f"{port}:8501"],
            "image": f"{shlex.quote(container)}:{shlex.quote(tag)}",
        },
    }

    return {
        "name": "explorer_app",
        "version": "3.8",
        "services": services,
    }


# type: ignore
def run_explorer_app(  # type: ignore  # noqa: PLR0913
    base_path: str,
    port: int = PORT,
    container: str = CONTAINER,
    output_path: str = OUTPUT_PATH,
    tag: t.Optional[str] = None,
    extra_volumes: t.Union[t.Optional[list], t.Optional[str]] = None,
    auth_provider: t.Optional[CloudCredentialsMount] = None,
):  # type: ignore
    """
    Run an Explorer App in a Docker container.

    Args:
      base_path: the base path where the Explorer App will be mounted.
      port: The port number to expose the Explorer App. Default is 8501.
      container: The name of the Docker container. Default is "fndnt/data_explorer".
      output_path: The path to the Docker Compose specification. Default is
       ".fondant/explorer-compose.yaml".
      tag: The tag/version of the Docker container. Default is "latest".
      extra_volumes: Extra volumes to mount in containers. You can use the --extra-volumes flag
      to specify extra volumes to mount in the containers this can be used:
        - to mount data directories to be used by the pipeline (note that if your pipeline's
            base_path is local it will already be mounted for you).
        - to mount cloud credentials
      auth_provider: The cloud provider to use for authentication. Default is None.
    """
    os.makedirs(".fondant", exist_ok=True)

    explorer_app_spec = _generate_explorer_spec(
        base_path=base_path,
        port=port,
        container=container,
        tag=tag,
        extra_volumes=extra_volumes,
        auth_provider=auth_provider,
    )

    with open(output_path, "w") as outfile:
        yaml.dump(explorer_app_spec, outfile, default_flow_style=False)

    cmd = [
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
    ]

    try:
        subprocess.check_call(cmd, stdout=subprocess.PIPE)  # nosec
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)

    logging.info(
        f"Running image from registry '{container}' with tag '{tag}' on port '{port}'",
    )
    logging.info(f"Access the explorer at http://localhost:{port}")


def stop_explorer_app(
    output_path: str = OUTPUT_PATH,
):
    """
    Stop the Explorer App.

    Args:
        output_path: The path to save the Docker Compose specification. Default is
        ".fondant/explorer-compose.yaml".
    """
    cmd = [
        "docker",
        "compose",
        "-f",
        output_path,
        "stop",
    ]

    try:
        subprocess.check_call(cmd, stdout=subprocess.PIPE)  # nosec
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)

    # check if the container is running
    logging.info(
        "Explorer app stopped successfully",
    )
