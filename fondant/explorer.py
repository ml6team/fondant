import logging
import shlex
import subprocess  # nosec
import typing as t

DEFAULT_CONTAINER = "ghcr.io/ml6team/data_explorer"
DEFAULT_TAG = "latest"
DEFAULT_PORT = 8501


def run_explorer_app(
    data_directory: t.Optional[str] = None,
    port: int = DEFAULT_PORT,
    credentials: str = "",
    container: str = DEFAULT_CONTAINER,
    tag: str = DEFAULT_TAG,
):
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

    # mount the local data directory to the container
    if data_directory:
        print(data_directory)
        cmd.extend(["-v", f"{shlex.quote(data_directory)}:/artifacts"])

    # add the image name
    cmd.extend(
        [
            f"{shlex.quote(container)}:{shlex.quote(tag)}",
        ],
    )

    logging.info(
        f"Running image from registry: {container} with tag: {tag} on port: {port}",
    )
    logging.info(f"Access the explorer at http://localhost:{port}")

    subprocess.call(cmd, stdout=subprocess.PIPE)  # nosec
