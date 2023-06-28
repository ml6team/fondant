# ruff: noqa: E501 (suppressing line length warnings in this file)
"""This file contains CLI script for the fondant package.


The entrypoint function is the main entrypoint for the CLI and is configured in the `pyproject.toml` file.

    [tool.poetry.scripts]
    script_name = "fondant.cli:entrypoint"

When installing the fondant package, the script will be available in the
environment.

eg `fondant --help`

If you want to extend the cli you can add a new subcommand by registering a new function in this file and adding it to the `entrypoint` function.
"""
import argparse
import importlib
import logging
import shutil
import sys
import textwrap

from fondant.compiler import DockerCompiler
from fondant.explorer import (
    DEFAULT_CONTAINER,
    DEFAULT_PORT,
    DEFAULT_TAG,
    run_explorer_app,
)
from fondant.pipeline import Pipeline
from fondant.runner import DockerRunner

logger = logging.getLogger(__name__)


def entrypoint():
    """Entrypoint for the fondant CLI."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Fondant is an Open-Source framework for building and running data pipelines.
        You can read more about fondant here: https://github.com/ml6team/fondant


        This CLI is used to interact with fondant pipelines like compiling and running your pipelines.

        Example:
        fondant compile my_project.my_pipeline.py:pipeline
        """,
        ),
        epilog=textwrap.dedent(
            """
        For a full list of commands run:
        fondant --help

        Or for a specific command run

        fondant <command> --help
        """,
        ),
    )
    subparsers = parser.add_subparsers()
    register_explore(subparsers)
    register_compile(subparsers)
    register_run(subparsers)

    sys.path.append(".")

    # display help if no arguments are provided
    args = parser.parse_args(sys.argv[1:] or ["--help"])

    args.func(args)


def register_explore(parent_parser):
    parser = parent_parser.add_parser(
        "explore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Explore and visualize the data produced by a fondant pipeline.

        This will spin up a  docker container that hosts a web application that allows you to explore the data produced by a fondant pipeline.

        The default address is http://localhost:8501. If the data that you want to explore is stored locally you can use the --data-directory flag to specify the path to the data.
        Alternatively you can use the --credentials flag to specify the path to a file that contains the credentials to access remote data (for S3, GCS, etc).

        Example:

        fondant explore -d my_project/data
        """,
        ),
    )
    parser.add_argument(
        "--data-directory",
        "-d",
        help="""Path to the source directory that contains the data produced
        by a fondant pipeline.""",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--container",
        "-r",
        default=DEFAULT_CONTAINER,
        help="Docker container to use. Defaults to ghcr.io/ml6team/data_explorer.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        default=DEFAULT_TAG,
        help="Docker image tag to use.",
    )
    parser.add_argument(
        "--port",
        "-p",
        default=DEFAULT_PORT,
        type=int,
        help="Port to expose the container on.",
    )
    parser.add_argument(
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
    )

    parser.set_defaults(func=explore)


def explore(args):
    if not args.data_directory:
        logging.warning(
            "You have not provided a data directory."
            + "To access local files, provide a local data directory"
            + " with the --data-directory flag.",
        )
    else:
        logging.info(f"Using data directory: {args.data_directory}")
        logging.info(
            "This directory will be mounted to /artifacts in the container.",
        )

    if not args.credentials:
        logging.warning(
            "You have not provided a credentials file. If you wish to access data "
            "from a cloud provider, mount the credentials file with the --credentials flag.",
        )

    if not shutil.which("docker"):
        logging.error(
            "Docker runtime not found. Please install Docker and try again.",
        )

    run_explorer_app(
        data_directory=args.data_directory,
        container=args.container,
        tag=args.tag,
        port=args.port,
        credentials=args.credentials,
    )


def register_compile(parent_parser):
    parser = parent_parser.add_parser(
        "compile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Compile a fondant pipeline into either a docker-compose.yml(local) or kubeflow spec file.

        The pipeline argument is a formatstring. The compiler will try to import the pipeline from the module specified in the formatstring.
        (NOTE: path is patched to include the current working directory so you can do relative imports)

        The --local or --kubeflow flag specifies the mode in which the pipeline will be compiled.
        You can use the --extra-volumes flag to specify extra volumes to mount in the containers this can be used:

        - to mount data directories to be used by the pipeline (note that if your pipeline's base_path is local it will already be mounted for you).
        - to mount cloud credentials (see examples))

        Example:
        fondant compile my_project.my_pipeline.py:pipeline --local --extra-volumes $HOME/.aws/credentials:/root/.aws/credentials

        fondant compile my_project.my_pipeline.py:pipeline --kubeflow --extra-volumes $HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json
        """,
        ),
    )
    parser.add_argument(
        "pipeline",
        help="Path to the fondant pipeline: path.to.module:instance",
        type=pipeline_from_string,
    )
    # add a mutually exclusive group for the mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--local", action="store_true")
    mode_group.add_argument("--kubeflow", action="store_true")

    parser.add_argument(
        "--output-path",
        "-o",
        help="Output directory",
        default="docker-compose.yml",
    )
    parser.add_argument(
        "--extra-volumes",
        help="Extra volumes to mount in containers",
        nargs="+",
    )

    parser.set_defaults(func=compile)


def compile(args):
    if args.local:
        compiler = DockerCompiler()
        compiler.compile(
            pipeline=args.pipeline,
            extra_volumes=args.extra_volumes,
            output_path=args.output_path,
        )
    elif args.kubeflow:
        msg = "Kubeflow compiler not implemented"
        raise NotImplementedError(msg)


def register_run(parent_parser):
    parser = parent_parser.add_parser(
        "run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Run a fondant pipeline locally or on kubeflow. The run command excepts a reference to an already compiled
        pipeline (see fondant compile --help for more info)
        OR a path to a spec file in which case it will compile the pipeline first and then run it.

        The --local or --kubeflow flag specifies the mode in which the pipeline will be ran.
        You can use the --extra-volumes flag to specify extra volumes to mount in the containers this can be used:

        Example:
        fondant run my_project.my_pipeline.py:pipeline --local --extra-volumes $HOME/.aws/credentials:/root/.aws/credentials
        fondant run ./my_compiled_kubeflow_pipeline.tgz --kubeflow
        """,
        ),
    )
    parser.add_argument(
        "ref",
        help="""Reference to the pipeline to run, can be a path to a spec file or
            a pipeline instance that will be compiled first""",
        action="store",
    )
    # add a mutually exclusive group for the mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--local", action="store_true")
    mode_group.add_argument("--kubeflow", action="store_true")

    parser.add_argument(
        "--output-path",
        "-o",
        help="Output directory",
        default="docker-compose.yml",
    )
    parser.add_argument(
        "--extra-volumes",
        help="Extra volumes to mount in containers",
        nargs="+",
    )
    parser.set_defaults(func=run)


def run(args):
    if args.local:
        try:
            pipeline = pipeline_from_string(args.ref)
        except ImportFromStringError:
            spec_ref = args.ref
        else:
            spec_ref = args.output_path
            logging.info(
                "Found reference to un-compiled pipeline... compiling to {spec_ref}",
            )
            compiler = DockerCompiler()
            compiler.compile(
                pipeline=pipeline,
                extra_volumes=args.extra_volumes,
                output_path=spec_ref,
            )
        finally:
            DockerRunner().run(spec_ref)
    elif args.kubeflow:
        msg = "Kubeflow runner not implemented"
        raise NotImplementedError(msg)


class ImportFromStringError(Exception):
    """Error raised when an import string is not valid."""


def pipeline_from_string(import_string: str) -> Pipeline:
    """Try to import a pipeline from a string otherwise raise an ImportFromStringError."""
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
