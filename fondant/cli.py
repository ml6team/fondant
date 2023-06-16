"""This file contains CLI script for the fondant package.

To add a script, add a function to this file and add it to `pyproject.toml` file
under the `[tool.poetry.scripts]` section.
To add a script, use the following format:

    [tool.poetry.scripts]
    script_name = "fondant.cli:script_function"

When installing the fondant package, the script will be available in the
environment.

e.g.

    'fondant-explorer --data-directory /path/to/data'
    'script_name --arg1 --arg2''
"""
import argparse
import logging
import shlex
import shutil
import subprocess  # nosec

DEFAULT_CONTAINER = "ghcr.io/ml6team/data_explorer"
DEFAULT_TAG = "latest"
DEFAULT_PORT = "8501"


def run_data_explorer():
    """Run the data explorer container."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run the data explorer container.")
    parser.add_argument(
        "--data-directory",
        "-d",
        help="Path to the source directory that contains the data produced by a fondant pipeline.",
    )
    parser.add_argument(
        "--container",
        "-r",
        default=DEFAULT_CONTAINER,
        help="Docker container to use. Defaults to ghcr.io/ml6team/data_explorer.",
    )
    parser.add_argument(
        "--tag", "-t", default=DEFAULT_TAG, help="Docker image tag to use."
    )
    parser.add_argument(
        "--port", "-p", default=DEFAULT_PORT, help="Port to expose the container on."
    )
    parser.add_argument(
        "--credentials",
        "-c",
        help="""Path mapping of the source (local) and target (docker file system) credential paths
             in the format of src:target
             \nExamples:\n
             Google Cloud: $HOME/.config/gcloud/application_default_credentials.json:/root/."
             + "config/gcloud/application_default_credentials.json
             AWS: HOME/.aws/credentials:/root/.aws/credentials
             More info on
             Google Cloud:
             https://cloud.google.com/docs/authentication/application-default-credentials
             AWS: https: // docs.aws.amazon.com/sdkref/latest/guide/file-location.html
        """,
    )
    args = parser.parse_args()

    if not args.data_directory:
        logging.warning(
            "You have not provided a data directory."
            + "To access local files, provide a local data directory"
            + " with the --data-directory flag."
        )
    else:
        logging.info(f"Using data directory: {args.data_directory}")
        logging.info("This directory will be mounted to /artifacts in the container.")

    if not shutil.which("docker"):
        logging.error("Docker runtime not found. Please install Docker and try again.")
        return

    cmd = [
        "docker",
        "run",
        "-p",
        f"{args.port}:8501",
    ]

    # mount the credentials file to the container
    if args.credentials:
        # check if path is read only
        if not args.credentials.endswith(":ro"):
            args.credentials += ":ro"

        cmd.extend(
            [
                "-v",
                args.credentials,
            ]
        )
    else:
        logging.warning(
            "You have not provided a credentials file. If you wish to access data "
            "from a cloud provider, mount the credentials file with the --credentials flag."
        )

    # mount the local data directory to the container
    if args.data_directory:
        cmd.extend(["-v", f"{shlex.quote(args.data_directory)}:/artifacts"])

    # add the image name
    cmd.extend(
        [
            f"{shlex.quote(args.container)}:{shlex.quote(args.tag)}",
        ]
    )

    logging.info(
        f"Running image from registry: {args.container} with tag: {args.tag} on port: {args.port}"
    )
    logging.info(f"Access the explorer at http://localhost:{args.port}")

    subprocess.call(cmd, stdout=subprocess.PIPE)  # nosec
