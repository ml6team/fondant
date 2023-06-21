"""This file contains CLI script for the fondant package.

To add a script, add a function to this file and add it to `pyproject.toml` file
under the `[tool.poetry.scripts]` section.
To add a script, use the following format:

    [tool.poetry.scripts]
    script_name = "fondant.cli:script_function"

When installing the fondant package, the script will be available in the
environment.

eg `fondant --help`
"""

import argparse
import logging
import shutil

from fondant.explorer import (
    DEFAULT_CONTAINER,
    DEFAULT_PORT,
    DEFAULT_TAG,
    run_explorer_app,
)

cli = argparse.ArgumentParser(description="Fondant CLI")
subparsers = cli.add_subparsers()


def entrypoint():
    args = cli.parse_args()
    args.func(args)


def argument(*name_or_flags, **kwargs):
    """Helper function to create an argument tuple for the subcommand decorator."""
    return ([*name_or_flags], kwargs)


def subcommand(name, parent_parser=subparsers, help=None, args=[]):
    """Decorator to add a subcommand to the CLI."""

    def decorator(func):
        parser = parent_parser.add_parser(name, help=help)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)

    return decorator


@subcommand(
    "explore",
    help="Explore a fondant pipeline",
    args=[
        argument(
            "--data-directory",
            "-d",
            help="""Path to the source directory that contains the data produced
            by a fondant pipeline.""",
            required=False,
        ),
        argument(
            "--container",
            "-r",
            default=DEFAULT_CONTAINER,
            help="Docker container to use. Defaults to ghcr.io/ml6team/data_explorer.",
        ),
        argument("--tag", "-t", default=DEFAULT_TAG, help="Docker image tag to use."),
        argument(
            "--port",
            "-p",
            default=DEFAULT_PORT,
            type=int,
            help="Port to expose the container on.",
        ),
        argument(
            "--credentials",
            "-c",
            help="""Path mapping of the source (local) and target (docker file system) 
             credential paths in the format of src:target
             \nExamples:\n
             Google Cloud: $HOME/.config/gcloud/application_default_credentials.json:/root/."
             + "config/gcloud/application_default_credentials.json
             AWS: HOME/.aws/credentials:/root/.aws/credentials
             More info on
             Google Cloud:
             https://cloud.google.com/docs/authentication/application-default-credentials
             AWS: https: // docs.aws.amazon.com/sdkref/latest/guide/file-location.html
        """,
        ),
    ],
)
def explore(args):
    """Defines the explore subcommand."""
    if not args.data_directory:
        logging.warning(
            "You have not provided a data directory."
            + "To access local files, provide a local data directory"
            + " with the --data-directory flag."
        )
    else:
        logging.info(f"Using data directory: {args.data_directory}")
        logging.info("This directory will be mounted to /artifacts in the container.")

    if not args.credentials:
        logging.warning(
            "You have not provided a credentials file. If you wish to access data "
            "from a cloud provider, mount the credentials file with the --credentials flag."
        )

    if not shutil.which("docker"):
        logging.error("Docker runtime not found. Please install Docker and try again.")
        return
    run_explorer_app(vars(args))
