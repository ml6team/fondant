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
import importlib
import logging
import shutil
import sys
import typing as t

from fondant.compiler import DockerCompiler
from fondant.explorer import (
    DEFAULT_CONTAINER,
    DEFAULT_PORT,
    DEFAULT_TAG,
    run_explorer_app,
)
from fondant.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
cli = argparse.ArgumentParser(description="Fondant CLI")
subparsers = cli.add_subparsers()


def entrypoint():
    sys.path.append(".")
    args = cli.parse_args()
    args.func(args)


def argument(*name_or_flags, **kwargs):
    """Helper function to create an argument tuple for the subcommand decorator."""
    return (list(name_or_flags), kwargs)


def distill_arguments(args: argparse.Namespace, remove: t.Optional[t.List[str]] = None):
    """Helper function to distill arguments to be passed on to the function."""
    args_dict = vars(args)
    args_dict.pop("func")
    if remove is not None:
        for arg in remove:
            args_dict.pop(arg)
    return args_dict


def subcommand(name, parent_parser=subparsers, help=None, args=None):
    """Decorator to add a subcommand to the CLI."""

    def decorator(func):
        parser = parent_parser.add_parser(name, help=help)
        if args is not None:
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
            type=str,
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
            + " with the --data-directory flag.",
        )
    else:
        logging.info(f"Using data directory: {args.data_directory}")
        logging.info("This directory will be mounted to /artifacts in the container.")

    if not args.credentials:
        logging.warning(
            "You have not provided a credentials file. If you wish to access data "
            "from a cloud provider, mount the credentials file with the --credentials flag.",
        )

    if not shutil.which("docker"):
        logging.error("Docker runtime not found. Please install Docker and try again.")

    function_args = distill_arguments(args)
    run_explorer_app(**function_args)


class ImportFromStringError(Exception):
    pass


def pipeline_from_string(import_string: str) -> Pipeline:
    module_str, _, attr_str = import_string.rpartition(":")
    if not attr_str or not module_str:
        raise ImportFromStringError(
            f"{import_string} is not a valid import string."
            + "Please provide a valid import string in the format of module:attr",
        )

    try:
        module = importlib.import_module(module_str)
    except ImportError:
        msg = f"{module_str} is not a valid module. Please provide a valid module."
        raise ImportFromStringError(
            msg,
        )

    try:
        for attr_str_element in attr_str.split("."):
            instance = getattr(module, attr_str_element)
    except AttributeError:
        msg = f"{attr_str} is not found in {module}."
        raise ImportFromStringError(msg)

    if not isinstance(instance, Pipeline):
        msg = f"{module}:{instance} is not a valid pipeline."
        raise ImportFromStringError(msg)

    return instance


@subcommand(
    "compile",
    help="Compile a fondant pipeline",
    args=[
        argument(
            "pipeline",
            help="Path to the fondant pipeline: path.to.module:instance",
            type=pipeline_from_string,
        ),
        argument(
            "--mode",
            "-m",
            help="Mode to run the pipeline in. Defaults to 'local'",
            default="local",
            choices=["local", "kubeflow"],
        ),
        argument(
            "--output-path",
            "-o",
            help="Output directory",
            default="docker-compose.yml",
        ),
        argument(
            "--extra-volumes",
            help="Extra volumes to mount in containers",
            nargs="+",
        ),
    ],
)
def compile(args):
    if args.mode == "local":
        compiler = DockerCompiler()
        function_args = distill_arguments(args, remove=["mode"])
        compiler.compile(**function_args)
    else:
        msg = "Kubeflow mode is not implemented yet."
        raise NotImplementedError(msg)
