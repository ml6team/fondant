import logging
import os
import shlex
import subprocess  # nosec
import typing as t
from pathlib import Path

from fsspec.implementations.local import LocalFileSystem

from fondant.filesystem import get_filesystem


# type: ignore
def run_explorer_app(  # type: ignore
    base_path: str,
    port: int,
    container: str,
    tag: str,
    credentials: t.Optional[str] = None,
):  # type: ignore
    """Run the data explorer container."""
    cmd = [
        "docker",
        "run",
        "--name",
        "fondant-explorer",
        "--rm",
        "-p",
        f"{port}:8501",
    ]

    # mount the credentials file to the container
    if credentials:
        # check if path is read only
        if not credentials.endswith(":ro"):
            credentials += ":ro"

        cmd.extend(
            [
                "-v",
                credentials,
            ],
        )

    if isinstance(get_filesystem(base_path), LocalFileSystem):
        logging.info(f"Using local base path: {base_path}")
        logging.info(
            "This directory will be mounted to /artifacts in the container.",
        )
        if not credentials:
            logging.warning(
                "You have not provided a credentials file. If you wish to access data "
                "from a cloud provider, mount the credentials file with the --credentials flag.",
            )
        data_directory_path = Path(base_path).resolve()
        host_machine_path = str(data_directory_path)
        container_path = os.path.join("/", data_directory_path.name)

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
        if credentials is None:
            raise RuntimeError(
                None,
                f"Specified base path `{base_path}`, Please provide valid credentials when using"
                f" a remote base path",
            )

        logging.info(f"Using remote base path: {base_path}")

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
