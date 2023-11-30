import logging
import os
import shlex
import subprocess  # nosec
import typing as t
from importlib.metadata import version
from pathlib import Path

import fsspec.core
from fsspec.implementations.local import LocalFileSystem


# type: ignore
def run_explorer_app(  # type: ignore
    base_path: str,
    port: int = 8501,
    container: str = "fndnt/data_explorer",
    tag: t.Optional[str] = None,
    extra_volumes: t.Union[t.Optional[list], t.Optional[str]] = None,
):  # type: ignore
    """
    Run an Explorer App in a Docker container.

    Args:
      base_path: the base path where the Explorer App will be mounted.
      port: The port number to expose the Explorer App. Default is 8501.
      container: The Docker container name or image to use. Default is "fndnt/data_explorer".
      tag: The tag/version of the Docker container. Default is "latest".
      extra_volumes: Extra volumes to mount in containers. You can use the --extra-volumes flag
      to specify extra volumes to mount in the containers this can be used:
        - to mount data directories to be used by the pipeline (note that if your pipeline's
            base_path is local it will already be mounted for you).
        - to mount cloud credentials
    """
    if tag is None:
        tag = version("fondant") if version("fondant") != "0.1.dev0" else "latest"

    if extra_volumes is None:
        extra_volumes = []

    if isinstance(extra_volumes, str):
        extra_volumes = [extra_volumes]

    cmd = [
        "docker",
        "run",
        "--pull",
        "always",
        "--name",
        "fondant-explorer",
        "--rm",
        "-p",
        f"{port}:8501",
    ]

    fs, _ = fsspec.core.url_to_fs(base_path)
    if isinstance(fs, LocalFileSystem):
        logging.info(f"Using local base path: {base_path}")
        logging.info(
            "This directory will be mounted to /artifacts in the container.",
        )
        data_directory_path = Path(base_path).resolve()
        host_machine_path = data_directory_path.as_posix()
        container_path = os.path.join("/", data_directory_path.name)

        # Mount extra volumes to the container
        if extra_volumes:
            for volume in extra_volumes:
                cmd.extend(
                    ["-v", volume],
                )

        # Mount the local base path to the container
        cmd.extend(
            ["-v", f"{shlex.quote(host_machine_path)}:{shlex.quote(container_path)}"],
        )

        # add the image name
        cmd.extend(
            [
                f"{shlex.quote(container)}:{shlex.quote(tag)}",
            ],
        )

        cmd.extend(
            ["--base_path", f"{container_path}"],
        )

    else:
        if not extra_volumes:
            raise RuntimeError(
                None,
                f"Specified base path `{base_path}`, Please provide valid credentials when using"
                f" a remote base path",
            )

        logging.info(f"Using remote base path: {base_path}")

        # Mount extra volumes to the container
        if extra_volumes:
            for volume in extra_volumes:
                cmd.extend(
                    ["-v", volume],
                )

        # add the image name
        cmd.extend(
            [
                f"{shlex.quote(container)}:{shlex.quote(tag)}",
            ],
        )

        # Add the remote base path as a container argument
        cmd.extend(
            ["--base_path", f"{base_path}"],
        )

    logging.info(
        f"Running image from registry: {container} with tag: {tag} on port: {port}",
    )
    logging.info(f"Access the explorer at http://localhost:{port}")

    subprocess.call(cmd, stdout=subprocess.PIPE)  # nosec
