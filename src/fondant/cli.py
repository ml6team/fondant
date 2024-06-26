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
import ast
import importlib
import inspect
import logging
import shutil
import sys
import textwrap
import typing as t
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

from fondant.core.schema import CloudCredentialsMount
from fondant.dataset import Dataset

if t.TYPE_CHECKING:
    from fondant.component import Component

logger = logging.getLogger(__name__)


def cloud_credentials_arg(value):
    try:
        return CloudCredentialsMount[value.upper()]
    except KeyError:
        msg = f"Invalid CloudCredentialsMount value: {value}"
        raise argparse.ArgumentTypeError(msg)


def entrypoint():
    """Entrypoint for the fondant CLI."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Fondant is an Open-Source framework for collaborative building of datasets.
        You can read more about fondant here: https://github.com/ml6team/fondant


        This CLI is used to interact with fondant datasets like compiling and running workflows to
        materialize datasets.

        Example:
        fondant run local my_dataset.py
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
    register_build(subparsers)
    register_execute(subparsers)
    register_compile(subparsers)
    register_run(subparsers)

    sys.path.append(".")

    # display help if no arguments are provided
    args, _ = parser.parse_known_args(sys.argv[1:] or ["--help"])
    if args.func.__name__ != "execute":
        args = parser.parse_args(sys.argv[1:] or ["--help"])

    args.func(args)


def register_explore(parent_parser):
    parser = parent_parser.add_parser(
        "explore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Explore and visualize the data of a Fondant dataset.

        This will spin up a docker container that hosts a web application that allows you to explore the dataset.

        The default address is http://localhost:8501. You can choose both a local and remote base path to explore. If the data that you want to explore is stored remotely, you
         should use the --extra-volumes flag to specify credentials or local files you  need to mount.

        Example:

        fondant explore start --base_path gs://foo/bar \
         -c $HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json
        """,
        ),
    )

    explore_subparser = parser.add_subparsers()
    start_parser = explore_subparser.add_parser(name="start", help="Start explorer app")
    stop_parser = explore_subparser.add_parser(name="stop", help="Stop explorer app")

    start_parser.add_argument(
        "--base_path",
        "-b",
        type=str,
        help="""Base path that contains the dataset (local or remote)
        .""",
    )
    start_parser.add_argument(
        "--container",
        "-r",
        type=str,
        default="fndnt/data_explorer",
        help="Docker container to use. Defaults to fndnt/data_explorer.",
    )
    start_parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default=version("fondant") if version("fondant") != "0.1.dev0" else "latest",
        help="Docker image tag to use.",
    )
    start_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8501,
        help="Port to expose the container on.",
    )
    start_parser.add_argument(
        "--output-path",
        type=str,
        default=".fondant/explorer-compose.yaml",
        help="The path to the Docker Compose specification.",
    )

    start_parser.add_argument(
        "--auth-provider",
        type=cloud_credentials_arg,
        choices=list(CloudCredentialsMount),
        help="Flag to authenticate with a cloud provider",
    )

    start_parser.add_argument(
        "--extra-volumes",
        help="""Extra volumes to mount in containers. You can use the --extra-volumes flag to specify extra volumes to mount in the containers this can be used:
        - to mount data directories to be used by the dataset (note that if your datasets base_path is local it will already be mounted for you).
        - to mount cloud credentials""",
        nargs="+",
    )

    stop_parser.add_argument(
        "--output-path",
        type=str,
        default=".fondant/explorer-compose.yaml",
        help="The path to the Docker Compose specification.",
    )

    start_parser.set_defaults(func=start_explore)
    stop_parser.set_defaults(func=stop_explore)


def start_explore(args):
    from fondant.explore import run_explorer_app

    if not shutil.which("docker"):
        logging.error(
            "Docker runtime not found. Please install Docker and try again.",
        )

    extra_volumes = []

    if args.extra_volumes:
        extra_volumes.extend(args.extra_volumes)

    run_explorer_app(
        base_path=args.base_path,
        container=args.container,
        tag=args.tag,
        port=args.port,
        extra_volumes=extra_volumes,
        auth_provider=args.auth_provider,
    )


def stop_explore(args):
    from fondant.explore import stop_explorer_app

    stop_explorer_app(
        output_path=args.output_path,
    )


def register_build(parent_parser):
    parser = parent_parser.add_parser(
        "build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Build a component and push it to the registry. The image name in the
        `fondant_component.yaml` will automatically be updated to use the new image.

        Example:

        fondant build components/my-component -tag my-tag
        """,
        ),
    )
    parser.add_argument(
        "component_dir",
        type=Path,
        help="""Path to the directory containing the component code, including a
        `fondant_component.yaml` and `Dockerfile`.""",
    )
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        help="Tag to add to built container. If the tag contains a `:`, it will be used as the "
        "full name for the image. If it does not contain a `:`, the image base name will be "
        "read from the `fondant_component.yaml` and combined into `base_name:tag`. If no tag is "
        "provided, the tag from the `fondant_component.yaml` is used directly.",
    )
    parser.add_argument(
        "--build-arg",
        action="append",
        help="Build arguments to pass to `docker build`. Format {key}={value}, can be repeated.",
        default=[],
    )
    parser.add_argument(
        "--nocache",
        action="store_true",
        help="Disable cache during building.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Downloads any updates to the FROM image in Dockerfiles.",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Name of the build-stage to build in a multi-stage Dockerfile.",
    )

    parser.add_argument(
        "--label",
        action="append",
        help="Label passed to `docker build` and assigned to the container. Format {key}={value}, can be repeated.",
        default=[],
    )

    parser.set_defaults(func=build)


def build(args):
    from fondant.build import build_component

    build_component(
        args.component_dir,
        tag=args.tag,
        build_args=args.build_arg,
        nocache=args.nocache,
        pull=args.pull,
        target=args.target,
        labels=args.label,
    )


def register_compile(parent_parser):
    parser = parent_parser.add_parser(
        "compile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Compile a fondant dataset into workflow specification file.

        The dataset argument is a formatstring. The compiler will try to import the dataset from the module specified in the formatstring.
        (NOTE: path is patched to include the current working directory so you can do relative imports)

        You can use different modes for fondant runners. Current existing modes are local and kubeflow.

        Examples of compiling component:
        fondant compile local --extra-volumes $HOME/.aws/credentials:/root/.aws/credentials my_project.my_dataset.py

        fondant compile kubeflow --extra-volumes $HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json my_project.my_dataset.py
        """,
        ),
    )

    compiler_subparser = parser.add_subparsers()

    local_parser = compiler_subparser.add_parser(name="local", help="Local compiler")
    local_parser.add_mutually_exclusive_group()

    kubeflow_parser = compiler_subparser.add_parser(
        name="kubeflow",
        help="Kubeflow compiler",
    )
    vertex_parser = compiler_subparser.add_parser(
        name="vertex",
        help="vertex compiler",
    )
    sagemaker_parser = compiler_subparser.add_parser(
        name="sagemaker",
        help="Sagemaker compiler",
    )

    # Local runner parser
    local_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, a path to a to a module containing
        the dataset instance that will be compiled (e.g. my-project/dataset.py)""",
        action="store",
    )
    local_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path of compiled workflow",
        default="docker-compose.yml",
    )
    local_parser.add_argument(
        "--extra-volumes",
        help="""Extra volumes to mount in containers. You can use the --extra-volumes flag to specify extra volumes to mount in the containers this can be used:
        - to mount data directories to be used by the workflow (note that if your dataset working directory is local it will already be mounted for you).
        - to mount cloud credentials""",
        nargs="+",
    )
    local_parser.add_argument(
        "--build-arg",
        action="append",
        help="Build arguments to pass to `docker build`. Format {key}={value}, can be repeated.",
        default=[],
    )

    local_parser.add_argument(
        "--auth-provider",
        type=cloud_credentials_arg,
        choices=list(CloudCredentialsMount),
        help="Flag to authenticate with a cloud provider",
    )

    # Kubeflow parser
    kubeflow_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, a path to a to a module containing
        the dataset instance that will be compiled (e.g. my-project/dataset.py)""",
        action="store",
    )
    kubeflow_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path of compiled dataset workflow",
        default="kubeflow-pipeline.yaml",
    )

    # vertex parser
    vertex_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, a path to a to a module containing
        the dataset instance that will be compiled (e.g. my-project/dataset.py)""",
        action="store",
    )
    vertex_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path of compiled workflow",
        default="vertex-pipeline.yml",
    )

    # sagemaker parser
    sagemaker_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, a path to a to a module containing
        the dataset instance that will be compiled (e.g. my-project/dataset.py)""",
        action="store",
    )
    sagemaker_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path of compiled workflow",
        default=".fondant/sagemaker_pipeline.json",
    )
    sagemaker_parser.add_argument(
        "--role-arn",
        help="""the Amazon Resource Name role to use for the processing steps""",
        default=None,
    )

    local_parser.set_defaults(func=compile_local)
    kubeflow_parser.set_defaults(func=compile_kfp)
    vertex_parser.set_defaults(func=compile_vertex)
    sagemaker_parser.set_defaults(func=compile_sagemaker)


def compile_local(args):
    from fondant.dataset.compiler import DockerCompiler

    extra_volumes = []

    if args.extra_volumes:
        extra_volumes.extend(args.extra_volumes)

    dataset = dataset_from_string(args.ref)
    compiler = DockerCompiler()
    compiler.compile(
        dataset=dataset,
        extra_volumes=extra_volumes,
        output_path=args.output_path,
        build_args=args.build_arg,
        auth_provider=args.auth_provider,
    )


def compile_kfp(args):
    from fondant.dataset.compiler import KubeFlowCompiler

    dataset = dataset_from_string(args.ref)
    compiler = KubeFlowCompiler()
    compiler.compile(dataset=dataset, output_path=args.output_path)


def compile_vertex(args):
    from fondant.dataset.compiler import VertexCompiler

    dataset = dataset_from_string(args.ref)
    compiler = VertexCompiler()
    compiler.compile(dataset=dataset, output_path=args.output_path)


def compile_sagemaker(args):
    from fondant.dataset.compiler import SagemakerCompiler

    dataset = dataset_from_string(args.ref)
    compiler = SagemakerCompiler()
    compiler.compile(
        dataset=dataset,
        output_path=args.output_path,
        role_arn=args.role_arn,
    )


def register_run(parent_parser):
    parser = parent_parser.add_parser(
        "run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Run a fondant dataset workflow locally or remote. The run command excepts a reference to an already compiled
        workflow (see fondant compile --help for more info)
        OR a path to a spec file in which case it will compile the dataset first and then run it.

        You can use different modes for fondant runners. Current existing modes are `local` and `kubeflow`.
        You can run `fondant <mode> --help` to find out more about the specific arguments for each mode.

        Examples of running component:
        fondant run local --auth-gcp
        fondant run kubeflow ./my_compiled_kubeflow_dataset.tgz
        """,
        ),
    )

    runner_subparser = parser.add_subparsers()

    local_parser = runner_subparser.add_parser(name="local", help="Local runner")

    kubeflow_parser = runner_subparser.add_parser(
        name="kubeflow",
        help="Kubeflow runner",
    )
    vertex_parser = runner_subparser.add_parser(
        name="vertex",
        help="Vertex runner",
    )
    sagemaker_parser = runner_subparser.add_parser(
        name="sagemaker",
        help="Sagemaker runner",
    )

    # Local runner parser
    local_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, can be a path to a spec file or
            a module containing the dataset instance that will be compiled first (e.g. dataset.py)
            """,
        action="store",
    )
    local_parser.add_argument(
        "--extra-volumes",
        nargs="+",
        help="""Extra volumes to mount in containers. You can use the --extra-volumes flag to specify extra volumes to mount in the containers this can be used:
        - to mount data directories to be used by the dataset (note that if your datasets working directory is local it will already be mounted for you).
        - to mount cloud credentials""",
    )

    local_parser.add_argument(
        "--working-directory",
        help="""Working directory where the dataset workflow will be executed.""",
    )

    local_parser.add_argument(
        "--build-arg",
        action="append",
        help="Build arguments for `docker build`",
    )

    local_parser.add_argument(
        "--auth-provider",
        type=cloud_credentials_arg,
        choices=list(CloudCredentialsMount),
        help="Flag to authenticate with a cloud provider",
    )

    # kubeflow runner parser
    kubeflow_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, can be a path to a spec file or
            a module containing the dataset instance that will be compiled first (e.g. dataset.py)
            """,
        action="store",
    )

    kubeflow_parser.add_argument(
        "--working-directory",
        help="""Working directory where the dataset workflow will be executed.""",
        required=True,
    )

    kubeflow_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path of compiled dataset workflow",
        default="kubeflow-pipeline.yaml",
    )
    kubeflow_parser.add_argument(
        "--host",
        help="KubeFlow host url",
        required=True,
    )

    # Vertex runner parser
    vertex_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, can be a path to a spec file or
            a module containing the dataset instance that will be compiled first (e.g. dataset.py)
            """,
        action="store",
    )
    vertex_parser.add_argument(
        "--working-directory",
        help="""Working directory where the dataset workflow will be executed.""",
        required=True,
    )
    vertex_parser.add_argument(
        "--project-id",
        help="""The project id of the GCP project used to submit the workflow""",
    )
    vertex_parser.add_argument(
        "--region",
        help="The region where to run the workflow",
    )

    vertex_parser.add_argument(
        "--output-path",
        "-o",
        help="Output path of compiled dataset",
        default="vertex-pipeline.yaml",
    )

    vertex_parser.add_argument(
        "--service-account",
        help="The service account used to launch jobs",
        default=None,
    )

    vertex_parser.add_argument(
        "--network",
        help="Network for the job to connect to, useful when peering with Vertex AI. Format "
        "should be 'projects/${project_number}/global/networks/${network}'",
        default=None,
    )

    # sagemaker runner parser
    sagemaker_parser.add_argument(
        "ref",
        help="""Reference to the dataset to materialize, can be a path to a spec file or
            a module containing the dataset instance that will be compiled first (e.g. dataset.py)
            """,
        action="store",
    )
    sagemaker_parser.add_argument(
        "--working-directory",
        help="""Working directory where the dataset workflow will be executed.""",
        required=True,
    )
    sagemaker_parser.add_argument(
        "--pipeline-name",
        help="""the name of the sagemaker pipeline to create""",
        default="fondant-pipeline",
    )

    sagemaker_parser.add_argument(
        "--role-arn",
        help="""the Amazon Resource Name role to use for the processing steps""",
        default=None,
    )

    local_parser.set_defaults(func=run_local)
    kubeflow_parser.set_defaults(func=run_kfp)
    vertex_parser.set_defaults(func=run_vertex)
    sagemaker_parser.set_defaults(func=run_sagemaker)


def run_local(args):
    from fondant.dataset.runner import DockerRunner

    extra_volumes = []

    working_directory = (
        args.working_directory if args.working_directory else "./artifacts"
    )

    if args.extra_volumes:
        extra_volumes.extend(args.extra_volumes)

    try:
        dataset = dataset_from_string(args.ref)
    except ModuleNotFoundError:
        dataset = args.ref

    runner = DockerRunner()
    runner.run(
        dataset=dataset,
        working_directory=working_directory,
        extra_volumes=extra_volumes,
        build_args=args.build_arg,
        auth_provider=args.auth_provider,
    )


def run_kfp(args):
    from fondant.dataset.runner import KubeflowRunner

    if not args.host:
        msg = "--host argument is required for running on Kubeflow"
        raise ValueError(msg)
    try:
        ref = dataset_from_string(args.ref)
    except ModuleNotFoundError:
        ref = args.ref

    runner = KubeflowRunner(host=args.host)

    runner.run(dataset=ref, working_directory=args.working_directory)


def run_vertex(args):
    from fondant.dataset.runner import VertexRunner

    try:
        ref = dataset_from_string(args.ref)
    except ModuleNotFoundError:
        ref = args.ref

    runner = VertexRunner(
        project_id=args.project_id,
        region=args.region,
        service_account=args.service_account,
        network=args.network,
    )

    runner.run(dataset=ref, working_directory=args.working_directory)


def run_sagemaker(args):
    from fondant.dataset.runner import SagemakerRunner

    try:
        ref = dataset_from_string(args.ref)
    except ModuleNotFoundError:
        ref = args.ref

    runner = SagemakerRunner()

    runner.run(
        dataset=ref,
        pipeline_name=args.dataset_name,
        role_arn=args.role_arn,
        working_directory=args.working_directory,
    )


def register_execute(parent_parser):
    parser = parent_parser.add_parser(
        "execute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        Execute a Fondant component using specified dataset parameters.

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

    parser.add_argument(
        "working-directory",
        help="""Working directory where the component will be executed""",
    )

    parser.set_defaults(func=execute)


def execute(args):
    from fondant.component.executor import ExecutorFactory

    component = component_from_module(args.ref)
    executor_factory = ExecutorFactory(component)
    executor = executor_factory.get_executor()
    executor.execute(component)


class ComponentImportError(Exception):
    """Error raised when an import string is not valid."""


class DatasetImportError(Exception):
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


def _called_with_wrong_args(f):
    """Check whether calling a function raised a ``TypeError`` because
    the call failed or because something in the factory raised the
    error.

    :param f: The function that was called.
    :return: ``True`` if the call failed.
    """
    tb = sys.exc_info()[2]

    try:
        while tb is not None:
            if tb.tb_frame.f_code is f.__code__:
                # In the function, it was called successfully.
                return False

            tb = tb.tb_next

        # Didn't reach the function.
        return True
    finally:
        # Delete tb to break a circular reference.
        # https://docs.python.org/2/library/sys.html#sys.exc_info
        del tb


def dataset_from_string(string_ref: str) -> Dataset:  # noqa: PLR0912
    """Get the workspace from the provided string reference.

    Inspired by Flask:
        https://github.com/pallets/flask/blob/d611989/src/flask/cli.py#L112

    Args:
        string_ref: String reference describing the dataset in the format {module}:{attribute}.
            The attribute can also be a function call, optionally including arguments:
            {module}:{function} or {module}:{function(args)}.

    Returns:
        The dataset obtained from the provided string
    """
    if ":" not in string_ref:
        return dataset_from_module(string_ref)

    module_str, dataset_str = string_ref.split(":")

    module = get_module(module_str)

    # Parse `dataset_str` as a single expression to determine if it's a valid
    # attribute name or function call.
    try:
        expr = ast.parse(dataset_str.strip(), mode="eval").body
    except SyntaxError:
        msg = f"Failed to parse {dataset_str} as an attribute name or function call."
        raise DatasetImportError(
            msg,
        ) from None

    if isinstance(expr, ast.Name):
        name = expr.id
        args = []
        kwargs = {}
    elif isinstance(expr, ast.Call):
        # Ensure the function name is an attribute name only.
        if not isinstance(expr.func, ast.Name):
            msg = f"Function reference must be a simple name: {dataset_str}."
            raise DatasetImportError(
                msg,
            )

        name = expr.func.id

        # Parse the positional and keyword arguments as literals.
        try:
            args = [ast.literal_eval(arg) for arg in expr.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}
        except ValueError:
            # literal_eval gives cryptic error messages, show a generic
            # message with the full expression instead.
            msg = f"Failed to parse arguments as literal values: {dataset_str}."
            raise DatasetImportError(
                msg,
            ) from None
    else:
        msg = f"Failed to parse {dataset_str} as an attribute name or function call."
        raise DatasetImportError(
            msg,
        )

    try:
        attr = getattr(module, name)
    except AttributeError as e:
        msg = f"Failed to find attribute {name} in {module.__name__}."
        raise DatasetImportError(
            msg,
        ) from e

    # If the attribute is a function, call it with any args and kwargs
    # to get the real dataset.
    if inspect.isfunction(attr):
        try:
            app = attr(*args, **kwargs)  # type: ignore
        except TypeError as e:
            if not _called_with_wrong_args(attr):
                raise

            msg = f"The factory {dataset_str} in module {module.__name__} could not be called with the specified arguments."
            raise DatasetImportError(
                msg,
            ) from e
    else:
        app = attr

    if isinstance(app, Dataset):
        return app

    msg = f"A valid Fondant workspace was not obtained from '{module.__name__}:{dataset_str}'."
    raise DatasetImportError(
        msg,
    )


def dataset_from_module(module_str: str) -> Dataset:
    """Try to import a dataset from a string otherwise raise an ImportFromStringError."""
    from fondant.dataset import Dataset

    module = get_module(module_str)

    dataset_instances = [
        obj for obj in module.__dict__.values() if isinstance(obj, Dataset)
    ]

    if not dataset_instances:
        msg = f"No dataset found in module {module_str}"
        raise DatasetImportError(msg)

    return dataset_instances[0]


def component_from_module(module_str: str) -> t.Type["Component"]:
    """Try to import a component from a module otherwise raise an ImportFromModuleError."""
    from fondant.component.component import BaseComponent

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
