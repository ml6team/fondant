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
import inspect
import logging
import shutil
import sys
import textwrap
import typing as t
from collections import defaultdict
from types import ModuleType

from fondant.compiler import DockerCompiler, KubeFlowCompiler
from fondant.component import BaseComponent, Component
from fondant.executor import ExecutorFactory
from fondant.explorer import (
    run_explorer_app,
)
from fondant.pipeline import Pipeline
from fondant.runner import DockerRunner, KubeflowRunner

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
        fondant compile my_project.my_pipeline.py
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
    register_execute(subparsers)
    register_compile(subparsers)
    register_run(subparsers)

    sys.path.append(".")

    # display help if no arguments are provided
    args, _ = parser.parse_known_args(sys.argv[1:] or ["--help"])
    if args.func.__name__ != "execute":
        args = parser.parse_args(sys.argv[1:] or ["--help"])

    args.func(args)


def set_default_output(args: argparse.Namespace):
    """Set the default output path depending on the runner type."""
    if args.output_path is None:
        if args.local:
            args.output_path = "docker-compose.yml"
        elif args.kubeflow:
            args.output_path = "pipeline.yaml"
        else:
            msg = "One of the arguments --local --kubeflow is required"
            raise ValueError(msg)

    return args


def register_explore(parent_parser):
    parser = parent_parser.add_parser(
        "explore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Explore and visualize the data produced by a fondant pipeline.

        This will spin up a docker container that hosts a web application that allows you to explore the data produced by a fondant pipeline.

        The default address is http://localhost:8501. You can choose both a local and remote base path to explore. If the data that you want to explore is stored remotely, you
         should use the --credentials flag to specify the path to a file that contains the credentials to access remote data (for S3, GCS, etc).

        Example:

        fondant explore --base_path gs://foo/bar \
         -c $HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json
        """,
        ),
    )
    parser.add_argument(
        "--base_path",
        "-b",
        type=str,
        help="""Base path that contains the data produced by a Fondant pipeline (local or remote)
        .""",
    )
    parser.add_argument(
        "--container",
        "-r",
        type=str,
        default="ghcr.io/ml6team/data_explorer",
        help="Docker container to use. Defaults to ghcr.io/ml6team/data_explorer.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default="latest",
        help="Docker image tag to use.",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8501,
        help="Port to expose the container on.",
    )
    parser.add_argument(
        "--credentials",
        "-c",
        type=str,
        default=None,
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
    if not shutil.which("docker"):
        logging.error(
            "Docker runtime not found. Please install Docker and try again.",
        )

    run_explorer_app(
        base_path=args.base_path,
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
        fondant compile my_project.my_pipeline.py --local --extra-volumes $HOME/.aws/credentials:/root/.aws/credentials

        fondant compile my_project.my_pipeline.py --kubeflow --extra-volumes $HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json
        """,
        ),
    )
    parser.add_argument(
        "ref",
        help="""Reference to the pipeline to run, can be a path to a spec file or
            a module containing the pipeline instance that will be compiled first (e.g. pipeline.py)
            """,
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
        default=None,
    )
    parser.add_argument(
        "--extra-volumes",
        help="Extra volumes to mount in containers",
        nargs="+",
    )
    parser.add_argument(
        "--build-arg",
        action="append",
        help="Build arguments to pass to `docker build`. Format {key}={value}.",
        default=[],
    )

    parser.set_defaults(func=compile)


def compile(args):
    args = set_default_output(args)
    pipeline = pipeline_from_module(args.ref)

    if args.local:
        compiler = DockerCompiler()
        compiler.compile(
            pipeline=pipeline,
            extra_volumes=args.extra_volumes,
            output_path=args.output_path,
            build_args=args.build_arg,
        )
    elif args.kubeflow:
        compiler = KubeFlowCompiler()
        compiler.compile(pipeline=pipeline, output_path=args.output_path)


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
        fondant run my_project.my_pipeline.py --local --extra-volumes $HOME/.aws/credentials:/root/.aws/credentials
        fondant run ./my_compiled_kubeflow_pipeline.tgz --kubeflow
        """,
        ),
    )
    parser.add_argument(
        "ref",
        help="""Reference to the pipeline to run, can be a path to a spec file or
            a module containing the pipeline instance that will be compiled first (e.g. pipeline.py)
            """,
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
        default=None,
    )
    parser.add_argument(
        "--extra-volumes",
        help="Extra volumes to mount in containers",
        nargs="+",
    )
    parser.add_argument(
        "--build-arg",
        action="append",
        help="Build arguments to pass to `docker build`. Format {key}={value}.",
    )
    parser.add_argument("--host", help="KubeFlow pipeline host url", required=False)
    parser.set_defaults(func=run)


def run(args):
    args = set_default_output(args)

    if args.local:
        try:
            pipeline = pipeline_from_module(args.ref)
        except ModuleNotFoundError:
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
                build_args=args.build_arg,
            )
        finally:
            DockerRunner().run(spec_ref)

    elif args.kubeflow:
        if not args.host:
            msg = "--host argument is required for running on Kubeflow"
            raise ValueError(msg)
        try:
            pipeline = pipeline_from_module(args.ref)
        except ModuleNotFoundError:
            spec_ref = args.ref
        else:
            spec_ref = args.output_path
            logging.info(
                f"Found reference to un-compiled pipeline... compiling to {spec_ref}",
            )
            compiler = KubeFlowCompiler()
            compiler.compile(pipeline=pipeline, output_path=spec_ref)
        finally:
            runner = KubeflowRunner(host=args.host)
            runner.run(input_spec=spec_ref)


def register_execute(parent_parser):
    parser = parent_parser.add_parser(
        "execute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Execute a Fondant component using specified pipeline parameters.

        This command is intended to be included in the entrypoint of a component's Dockerfile. The provided argument
        to this command should indicate the module where the component's implementation resides.

        The command attempts to import the user-implemented component from the specified module and
        then executes it with the user-provided arguments.

        Example:

        fondant execute main.py
        """,
        ),
    )
    parser.add_argument(
        "ref",
        help="""Reference to the module containing the component to run""",
        action="store",
    )

    parser.set_defaults(func=execute)


def execute(args):
    component = component_from_module(args.ref)
    executor_factory = ExecutorFactory(component)
    executor = executor_factory.get_executor()
    executor.execute(component)


class ComponentImportError(Exception):
    """Error raised when an import string is not valid."""


class PipelineImportError(Exception):
    """Error raised when an import from module is not valid."""


def get_module(module_str: str) -> ModuleType:
    """Function that retrieves module from a module string."""
    if ".py" in module_str:
        module_str = module_str.rsplit(".py", 1)[0]

    module_str = module_str.replace("/", ".")

    try:
        module = importlib.import_module(module_str)
    except ModuleNotFoundError:
        msg = f"`{module_str}` was not found. Please provide a valid module."
        raise ModuleNotFoundError(msg)

    return module


def pipeline_from_module(module_str: str) -> Pipeline:
    """Try to import a pipeline from a string otherwise raise an ImportFromStringError."""
    module = get_module(module_str)

    pipeline_instances = [
        obj for obj in module.__dict__.values() if isinstance(obj, Pipeline)
    ]

    if not pipeline_instances:
        msg = f"No pipeline found in module {module_str}"
        raise PipelineImportError(msg)

    if len(pipeline_instances) > 1:
        msg = (
            f"Found multiple instantiated pipelines in {module_str}. Only one pipeline "
            f"can be present"
        )
        raise PipelineImportError(msg)

    pipeline = pipeline_instances[0]
    logger.info(f"Pipeline `{pipeline.name}` found in module {module_str}")

    return pipeline


def component_from_module(module_str: str) -> t.Type[Component]:
    """Try to import a component from a module otherwise raise an ImportFromModuleError."""
    module = get_module(module_str)
    class_members = inspect.getmembers(module, inspect.isclass)

    component_classes_dict = defaultdict(list)

    for name, cls in class_members:
        if issubclass(cls, BaseComponent):
            order = len(cls.__mro__)
            component_classes_dict[order].append((name, cls))

    if len(component_classes_dict) == 0:
        msg = f"No Component found in module {module_str}"
        raise ComponentImportError(msg)

    max_order = max(component_classes_dict)
    found_components = component_classes_dict[max_order]

    if len(found_components) > 1:
        msg = (
            f"Found multiple components in {module_str}: {found_components}. Only one component "
            f"can be present"
        )
        raise ComponentImportError(msg)

    component_name, component_cls = found_components[0]
    logger.info(f"Component `{component_name}` found in module {module_str}")

    return component_cls
