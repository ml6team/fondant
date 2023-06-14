"""This file contains CLI script for the fondant package.

To add a script, add a function to this file and add it to `pyproject.toml` file
under the `[tool.poetry.scripts]` section.
To add a script, use the following format:

    [tool.poetry.scripts]
    script_name = "fondant.cli:script_function"

When installing the fondant package, the script will be available in the
environment.

e.g.

    'fondant-explorer --source /path/to/data'
    'script_name --arg1 --arg2''
"""
import argparse
import logging
import shlex
import shutil
import subprocess  # nosec

DEFAULT_REGISTRY = "ghcr.io/ml6team/data_explorer"
DEFAULT_TAG = "latest"
DEFAULT_PORT = "8501"


def run_data_explorer():
    """Run the data explorer container."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run the data explorer container.")
    parser.add_argument("--source", "-s", help="Path to the source directory.")
    parser.add_argument(
        "--registry", "-r", default=DEFAULT_REGISTRY, help="Docker registry to use."
    )
    parser.add_argument(
        "--tag", "-t", default=DEFAULT_TAG, help="Docker image tag to use."
    )
    parser.add_argument(
        "--port", "-p", default=DEFAULT_PORT, help="Port to expose the container on."
    )
    args = parser.parse_args()

    if not args.source:
        logging.error(
            "Please provide a source directory with the --source or -s option."
        )
        return

    if not shutil.which("docker"):
        logging.error("Docker runtime not found. Please install Docker and try again.")
        return

    cmd = [
        "docker",
        "run",
        "-p",
        f"{args.port}:8501",
        "--mount",
        f"type=bind,source={shlex.quote(args.source)},target=/artifacts",
        f"{shlex.quote(args.registry)}:{shlex.quote(args.tag)}",
    ]

    logging.info(
        f"Running image from registry: {args.registry} with tag: {args.tag} on port: {args.port}"
    )
    logging.info(f"Access the explorer at http://localhost:{args.port}")

    subprocess.call(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)  # nosec
