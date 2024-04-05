import json
import logging
import os
import re
import shlex
import tempfile
import textwrap
import typing as t
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import yaml
from fsspec.registry import known_implementations

from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidDatasetDefinition
from fondant.core.manifest import Metadata
from fondant.core.schema import CloudCredentialsMount, DockerVolume
from fondant.dataset import (
    VALID_ACCELERATOR_TYPES,
    VALID_VERTEX_ACCELERATOR_TYPES,
    Dataset,
    Image,
)

logger = logging.getLogger(__name__)

# export DASK_DIAGNOSTICS_PORT="" to get a dynamic port assigned
DASK_DIAGNOSTICS_PORT = os.environ.get("DASK_DIAGNOSTICS_PORT", ":8787")

KubeflowCommandArguments = t.List[t.Union[str, t.Dict[str, str]]]


class Compiler(ABC):
    """Abstract base class for a compiler."""

    @abstractmethod
    def compile(self, *args, **kwargs) -> None:
        """Abstract method to invoke compilation."""

    @abstractmethod
    def _set_configuration(self, *args, **kwargs) -> None:
        """Abstract method to set pipeline configuration."""

    def log_unused_configurations(self, **kwargs):
        """Log configurations that are set but will be unused."""
        for config_name, config_value in kwargs.items():
            if config_value is not None:
                logger.warning(
                    f"Configuration `{config_name}` is set with `{config_value}` but has no effect"
                    f" for runner `{self.__class__.__name__}`.",
                )

    @staticmethod
    def _build_entrypoint(image: Image) -> t.List[str]:
        """Build the entrypoint to execute the provided image."""
        if not image.script:
            # Not a lightweight python component
            return ["fondant", "execute", "main"]

        command = ""
        if image.extra_requires:
            requirements = "\n".join(image.extra_requires)
            command += textwrap.dedent(
                f"""\
                printf {shlex.quote(requirements)} > 'requirements.txt'
                python3 -m pip install -r requirements.txt
            """,
            )

        command += textwrap.dedent(
            f"""\
            printf {shlex.quote(image.script)} > 'main.py'
            fondant execute main "$@"
        """,
        )

        return [
            "sh",
            "-ec",
            command,
            "--",  # All arguments provided after this will be passed to `fondant execute main`
        ]


class DockerCompiler(Compiler):
    """Compiler that creates a docker-compose spec from a dataset."""

    def compile(
        self,
        dataset: Dataset,
        *,
        working_directory: str,
        output_path: str = "docker-compose.yml",
        extra_volumes: t.Union[t.Optional[list], t.Optional[str]] = None,
        build_args: t.Optional[t.List[str]] = None,
        auth_provider: t.Optional[CloudCredentialsMount] = None,
    ) -> None:
        """Compile a dataset workflow to docker-compose spec and save it to a specified output path.

        Args:
            dataset: the dataset to compile
            working_directory: working_directory to store local dataset artifacts
            output_path: the path where to save the docker-compose spec
            extra_volumes: a list of extra volumes (using the Short syntax:
              https://docs.docker.com/compose/compose-file/05-services/#short-syntax-5)
              to mount in the docker-compose spec.
            build_args: List of build arguments to pass to docker
            auth_provider: The cloud provider to use for authentication. Default is None.

        """
        if extra_volumes is None:
            extra_volumes = []

        if isinstance(extra_volumes, str):
            extra_volumes = [extra_volumes]

        if auth_provider:
            extra_volumes.append(auth_provider.get_path())

        spec = self._generate_spec(
            dataset,
            working_directory=working_directory,
            extra_volumes=extra_volumes,
            build_args=build_args or [],
        )

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        with open(output_path, "w") as outfile:
            yaml.dump(spec, outfile, Dumper=NoAliasDumper, default_flow_style=False)

        logger.info(f"Successfully compiled to {output_path}")

    @staticmethod
    def _patch_path(base_path: str) -> t.Tuple[str, t.Optional[DockerVolume]]:
        """Helper that checks if the base_path is local or remote,
        if local it patches the base_path and prepares a bind mount
        Returns a tuple containing the path and volume.
        """

        def is_remote_path(path: Path) -> bool:
            """Check if the path is remote."""
            scheme = urlparse(str(path)).scheme

            fsspec_schemes = set(known_implementations.keys()) - {"local", "file"}

            return scheme in fsspec_schemes

        def resolve_local_base_path(base_path: Path) -> Path:
            """Resolve local base path and create base directory if it no exist."""
            p_base_path = base_path.resolve()
            try:
                if p_base_path.exists():
                    logger.info(
                        f"Base path found on local system, setting up {base_path} as mount volume",
                    )
                else:
                    p_base_path.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        f"Base path not found on local system, created base path and setting up "
                        f"{base_path} as mount volume",
                    )
            except Exception as e:
                msg = f"Unable to create and mount local base path. {e}"
                raise ValueError(msg)

            return p_base_path

        p_base_path = Path(base_path)
        if is_remote_path(p_base_path):
            logger.info(f"Base path {base_path} is remote")
            return base_path, None

        p_base_path = resolve_local_base_path(p_base_path)
        volume = DockerVolume(
            type="bind",
            source=str(p_base_path),
            target=f"/{p_base_path.stem}",
        )
        path = f"/{p_base_path.stem}"
        return path, volume

    def _generate_spec(
        self,
        dataset: Dataset,
        working_directory: str,
        *,
        extra_volumes: t.List[str],
        build_args: t.List[str],
    ) -> dict:
        """Generate a docker-compose spec as a python dictionary,
        loops over the dataset graph to create services and their dependencies.
        """
        path, volume = self._patch_path(base_path=working_directory)
        run_id = dataset.manifest.run_id

        services = {}

        dataset.validate()

        component_cache_key = None

        for component_id, component in dataset._graph.items():
            component_op = component["operation"]

            component_cache_key = component_op.get_component_cache_key(
                previous_component_cache=component_cache_key,
            )

            # Generate default values for manifest and dataset location based on working_dir
            manifest_location = (
                f"{working_directory}/{dataset.name}/{run_id}/{component_id}"
                f"/manifest.json"
            )

            metadata = Metadata(
                dataset_name=dataset.name,
                run_id=run_id,
                component_id=component_id,
                cache_key=component_cache_key,
                manifest_location=manifest_location,
            )

            logger.info(f"Compiling service for {component_id}")

            entrypoint = self._build_entrypoint(component_op.image)

            # add metadata argument to command
            command = ["--metadata", metadata.to_json()]

            # add in and out manifest paths to command
            command.extend(
                [
                    "--output_manifest_path",
                    f"{path}/{metadata.dataset_name}/{metadata.run_id}/"
                    f"{component_id}/manifest.json",
                ],
            )

            # Add working directory to command
            command.extend(
                [
                    "--working_directory",
                    working_directory,
                ],
            )

            # add arguments if any to command
            for key, value in component_op.arguments.items():
                if isinstance(value, (dict, list)):
                    command.extend([f"--{key}", json.dumps(value)])
                else:
                    command.extend([f"--{key}", f"{value}"])

            # resolve dependencies
            depends_on = {}
            if component["dependencies"]:
                for dependency in component["dependencies"]:
                    depends_on[dependency] = {
                        "condition": "service_completed_successfully",
                    }
                    # there is an input manifest if the component has dependencies, use the manifest
                    # from the previous component
                    command.extend(
                        [
                            "--input_manifest_path",
                            f"{path}/{metadata.dataset_name}/{metadata.run_id}/"
                            f"{dependency}/manifest.json",
                        ],
                    )

            elif dataset.manifest.contains_data():
                command.extend(
                    [
                        "--input_manifest_path",
                        f"{dataset.manifest.manifest_location}",
                    ],
                )

            volumes: t.List[t.Union[str, dict]] = []
            if volume:
                volumes.append(asdict(volume))
            if extra_volumes:
                volumes.extend(extra_volumes)

            ports = [f"8787{DASK_DIAGNOSTICS_PORT}"]

            services[component_id] = {
                "entrypoint": entrypoint,
                "command": command,
                "depends_on": depends_on,
                "volumes": volumes,
                "ports": ports,
                "labels": {
                    "dataset_description": dataset.description,
                },
            }

            self._set_configuration(services, component_op, component_id)

            if component_op.dockerfile_path is not None:
                logger.info(
                    f"Found Dockerfile for {component_id}, adding build step.",
                )
                services[component_id]["build"] = {
                    "context": str(component_op.component_dir.absolute()),
                    "args": build_args,
                }
            else:
                services[component_id]["image"] = component_op.component_spec.image

        return {
            "name": dataset.name,
            "version": "3.8",
            "services": services,
        }

    def _set_configuration(self, services, fondant_component_operation, component_id):
        resources_dict = fondant_component_operation.resources.to_dict()

        accelerator_name = resources_dict.pop("accelerator_name")
        accelerator_number = resources_dict.pop("accelerator_number")

        # Unused configurations
        self.log_unused_configurations(**resources_dict)

        if accelerator_name is not None:
            if accelerator_name not in VALID_ACCELERATOR_TYPES:
                msg = (
                    f"Configured accelerator `{accelerator_name}`"
                    f" is not a valid accelerator type for Docker Compose compiler."
                    f" Available options: {VALID_VERTEX_ACCELERATOR_TYPES}"
                )
                raise InvalidDatasetDefinition(msg)

            if accelerator_name == "GPU":
                services[component_id]["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": accelerator_number,
                                    "capabilities": ["gpu"],
                                },
                            ],
                        },
                    },
                }
            elif accelerator_name == "TPU":
                msg = "TPU configuration is not yet implemented for Docker Compose "
                raise NotImplementedError(msg)

        return services


class KubeflowComponentSpec:
    """
    Class representing a Kubeflow component specification.

    Args:
        specification: The kubeflow component specification as a Python dict
    """

    def __init__(self, specification: t.Dict[str, t.Any]) -> None:
        self._specification = specification

    @staticmethod
    def convert_arguments(fondant_component: ComponentSpec):
        args = {}
        for arg in fondant_component.args.values():
            arg_type_dict = {}

            # Enable isOptional attribute in spec if arg is Optional and defaults to None
            if arg.optional and arg.default is None:
                arg_type_dict["isOptional"] = True
            if arg.default is not None:
                arg_type_dict["defaultValue"] = arg.default

            args[arg.name] = {
                "parameterType": arg.kubeflow_type,
                "description": arg.description,
                **arg_type_dict,  # type: ignore
            }

        return args

    @classmethod
    def from_fondant_component_spec(
        cls,
        fondant_component: ComponentSpec,
        command: t.List[str],
        image_uri: str,
    ):
        """Generate a Kubeflow component spec from a Fondant component spec."""
        input_definitions = {
            "parameters": {
                **cls.convert_arguments(fondant_component),
            },
        }

        kfp_safe_name = (
            re.sub(
                "-+",
                "-",
                re.sub("[^-0-9a-z]+", "-", fondant_component.safe_name.lower()),
            )
            .lstrip("-")
            .rstrip("-")
        )
        specification = {
            "components": {
                "comp-"
                + kfp_safe_name: {
                    "executorLabel": "exec-" + kfp_safe_name,
                    "inputDefinitions": input_definitions,
                },
            },
            "deploymentSpec": {
                "executors": {
                    "exec-"
                    + kfp_safe_name: {
                        "container": {
                            "command": command,
                            "image": image_uri,
                        },
                    },
                },
            },
            "pipelineInfo": {"name": kfp_safe_name},
            "root": {
                "dag": {
                    "tasks": {
                        kfp_safe_name: {
                            "cachingOptions": {"enableCache": True},
                            "componentRef": {"name": "comp-" + kfp_safe_name},
                            "inputs": {
                                "parameters": {
                                    param: {"componentInputParameter": param}
                                    for param in input_definitions["parameters"]
                                },
                            },
                            "taskInfo": {"name": kfp_safe_name},
                        },
                    },
                },
                "inputDefinitions": input_definitions,
            },
            "schemaVersion": "2.1.0",
            "sdkVersion": "kfp-2.6.0",
        }
        return cls(specification)

    def to_file(self, path: t.Union[str, Path]) -> None:
        """Dump the component specification to the file specified by the provided path."""
        with open(path, "w", encoding="utf-8") as file_:
            yaml.dump(
                self._specification,
                file_,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    def to_string(self) -> str:
        """Return the component specification as a string."""
        return json.dumps(self._specification)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"


class KubeFlowCompiler(Compiler):
    """Compiler that creates a Kubeflow pipeline spec from a dataset."""

    def __init__(self):
        self._resolve_imports()

    def _resolve_imports(self):
        """Resolve imports for the Kubeflow compiler."""
        try:
            import kfp
            import kfp.kubernetes as kfp_kubernetes

            self.kfp = kfp
            self.kfp_kubernetes = kfp_kubernetes

        except ImportError:
            msg = """You need to install kfp to use the Kubeflow compiler,\n
                     you can install it with `pip install fondant[kfp]`"""
            raise ImportError(
                msg,
            )

    def compile(
        self,
        dataset: Dataset,
        working_directory: str,
        output_path: str,
    ) -> None:
        """Compile a dataset workflow to Kubeflow pipeline spec and save it to a specified
        output path.

        Args:
            dataset: the dataset to compile
            working_directory: path of the working directory
            output_path: the path where to save the Kubeflow pipeline spec
        """
        run_id = dataset.manifest.run_id
        dataset.validate()
        logger.info(f"Compiling {dataset.name} to {output_path}")

        def set_component_exec_args(
            *,
            component_op,
            component_args: t.List[str],
            input_manifest_path: bool,
        ):
            """Dump Fondant specification arguments to kfp command executor arguments."""
            dumped_args: KubeflowCommandArguments = []

            component_args.extend(
                [
                    "output_manifest_path",
                    "metadata",
                    "working_directory",
                ],
            )
            if input_manifest_path:
                component_args.append("input_manifest_path")

            for arg in component_args:
                arg_name = arg.strip().replace(" ", "_")
                arg_name_cmd = f"--{arg_name}"

                dumped_args.append(arg_name_cmd)
                dumped_args.append("{{$.inputs.parameters['" + f"{arg_name}" + "']}}")

            component_op.component_spec.implementation.container.args = dumped_args

            return component_op

        @self.kfp.dsl.pipeline(name=dataset.name, description=dataset.description)
        def kfp_pipeline():
            previous_component_task = None
            component_cache_key = None

            for component_name, component in dataset._graph.items():
                logger.info(f"Compiling service for {component_name}")

                component_op = component["operation"]
                # convert ComponentOp to Kubeflow component
                command = self._build_entrypoint(component_op.image)
                image_uri = component_op.image.base_image
                kubeflow_spec = KubeflowComponentSpec.from_fondant_component_spec(
                    component_op.component_spec,
                    command=command,
                    image_uri=image_uri,
                )

                kubeflow_component_op = self.kfp.components.load_component_from_text(
                    text=kubeflow_spec.to_string(),
                )

                # Remove None values from arguments
                component_args = {
                    k: v for k, v in component_op.arguments.items() if v is not None
                }

                component_cache_key = component_op.get_component_cache_key(
                    previous_component_cache=component_cache_key,
                )
                metadata = Metadata(
                    run_id=run_id,
                    component_id=component_name,
                    cache_key=component_cache_key,
                    dataset_name="dataset",
                    manifest_location=f"{working_directory}/{dataset.name}/{run_id}/{component_name}/manifest.json",
                )

                output_manifest_path = (
                    f"{working_directory}/{metadata.dataset_name}/{metadata.run_id}/"
                    f"{metadata.component_id}/manifest.json"
                )
                # Set the execution order of the component task to be after the previous
                # component task.
                if component["dependencies"]:
                    for dependency in component["dependencies"]:
                        input_manifest_path = (
                            f"{working_directory}/{metadata.dataset_name}/{metadata.run_id}"
                            f"/{dependency}/manifest.json"
                        )
                        kubeflow_component_op = set_component_exec_args(
                            component_op=kubeflow_component_op,
                            component_args=list(component_args.keys()),
                            input_manifest_path=True,
                        )
                        component_task = kubeflow_component_op(
                            input_manifest_path=input_manifest_path,
                            output_manifest_path=output_manifest_path,
                            working_directory=working_directory,
                            metadata=metadata.to_json(),
                            **component_args,
                        )
                        component_task.after(previous_component_task)

                else:
                    kubeflow_component_op = set_component_exec_args(
                        component_op=kubeflow_component_op,
                        component_args=list(component_args.keys()),
                        input_manifest_path=False,
                    )
                    component_task = kubeflow_component_op(
                        metadata=metadata.to_json(),
                        output_manifest_path=output_manifest_path,
                        working_directory=working_directory,
                        **component_args,
                    )

                # Set optional configuration
                component_task = self._set_configuration(
                    component_task,
                    component_op,
                )

                # Disable caching
                component_task.set_caching_options(enable_caching=False)

                previous_component_task = component_task

        logger.info(f"Compiling {dataset.name} to {output_path}")

        self.kfp.compiler.Compiler().compile(kfp_pipeline, output_path)  # type: ignore
        logger.info("Pipeline compiled successfully")

    def _set_configuration(self, task, fondant_component_operation):
        # Used configurations
        resources_dict = fondant_component_operation.resources.to_dict()

        accelerator_number = resources_dict.pop("accelerator_number")
        accelerator_name = resources_dict.pop("accelerator_name")
        node_pool_label = resources_dict.pop("node_pool_label")
        node_pool_name = resources_dict.pop("node_pool_name")
        cpu_request = resources_dict.pop("cpu_request")
        cpu_limit = resources_dict.pop("cpu_limit")
        memory_request = resources_dict.pop("memory_request")
        memory_limit = resources_dict.pop("memory_limit")

        # Unused configurations
        self.log_unused_configurations(**resources_dict)

        # Assign optional specification
        if cpu_request is not None:
            task.set_memory_request(cpu_request)
        if cpu_limit is not None:
            task.set_memory_limit(cpu_limit)
        if memory_request is not None:
            task.set_memory_request(memory_request)
        if memory_limit is not None:
            task.set_memory_limit(memory_limit)
        if accelerator_name is not None:
            if accelerator_name not in VALID_ACCELERATOR_TYPES:
                msg = (
                    f"Configured accelerator `{accelerator_name}` is not a valid accelerator type"
                    f"for Kubeflow compiler. Available options: {VALID_ACCELERATOR_TYPES}"
                )
                raise InvalidDatasetDefinition(msg)

            task.set_accelerator_limit(accelerator_number)
            if accelerator_name == "GPU":
                task.set_accelerator_type("nvidia.com/gpu")
            elif accelerator_name == "TPU":
                task.set_accelerator_type("cloud-tpus.google.com/v3")
        if node_pool_name is not None and node_pool_label is not None:
            task = self.kfp_kubernetes.add_node_selector(
                task,
                node_pool_label,
                node_pool_name,
            )
        return task


class VertexCompiler(KubeFlowCompiler):
    def __init__(self):
        super().__init__()
        self.resolve_imports()

    def resolve_imports(self):
        """Resolve imports for the Vertex compiler."""
        try:
            import kfp

            self.kfp = kfp

        except ImportError:
            msg = """You need to install kfp to use the Vertex compiler,\n
                     you can install it with `pip install fondant[vertex]`"""
            raise ImportError(
                msg,
            )

    def _set_configuration(self, task, fondant_component_operation):
        # Used configurations
        resources_dict = fondant_component_operation.resources.to_dict()

        cpu_limit = resources_dict.pop("cpu_limit")
        memory_limit = resources_dict.pop("memory_limit")
        accelerator_number = resources_dict.pop("accelerator_number")
        accelerator_name = resources_dict.pop("accelerator_name")

        # Unused configurations
        self.log_unused_configurations(**resources_dict)

        # Assign optional specification
        if cpu_limit is not None:
            task.set_cpu_limit(cpu_limit)
        if memory_limit is not None:
            task.set_memory_limit(memory_limit)
        if accelerator_number is not None:
            task.set_accelerator_limit(accelerator_number)
            if accelerator_name not in VALID_VERTEX_ACCELERATOR_TYPES:
                msg = (
                    f"Configured accelerator `{accelerator_name}` is not a valid accelerator type"
                    f"for Vertex compiler. Available options: {VALID_VERTEX_ACCELERATOR_TYPES}"
                )
                raise InvalidDatasetDefinition(msg)

            task.set_accelerator_type(accelerator_name)

        return task


class SagemakerCompiler(Compiler):  # pragma: no cover
    def __init__(self):
        self.ecr_namespace = "fndnt-mirror"
        self._resolve_imports()

    def _resolve_imports(self):
        try:
            import boto3
            import sagemaker
            import sagemaker.processing
            import sagemaker.workflow.pipeline
            import sagemaker.workflow.steps

            self.boto3 = boto3
            self.sagemaker = sagemaker

        except ImportError:
            msg = """You need to install the sagemaker extras to use the sagemaker compiler,\n
                     you can install it with `pip install fondant[sagemaker]`"""
            raise ImportError(
                msg,
            )

    def _build_command(
        self,
        metadata: Metadata,
        working_directory: str,
        arguments: t.Dict[str, t.Any],
        dependencies: t.List[str] = [],
    ) -> t.List[str]:
        # add metadata argument to command
        command = ["--metadata", f"'{metadata.to_json()}'"]

        # add in and out manifest paths to command
        command.extend(
            [
                "--output_manifest_path",
                f"{working_directory}/{metadata.dataset_name}/{metadata.run_id}/"
                f"{metadata.component_id}/manifest.json",
            ],
        )

        # add arguments if any to command
        for key, value in arguments.items():
            if isinstance(value, (dict, list)):
                command.extend([f"--{key}", f"'{json.dumps(value)}'"])
            else:
                command.extend([f"--{key}", f"'{value}'"])

        # resolve dependencies
        if dependencies:
            for dependency in dependencies:
                # there is only an input manifest if the component has dependencies
                command.extend(
                    [
                        "--input_manifest_path",
                        f"{working_directory}/{metadata.dataset_name}/{metadata.run_id}/"
                        f"{dependency}/manifest.json",
                    ],
                )

        command.extend(
            [
                "--working_directory",
                working_directory,
            ],
        )

        return command

    def _check_ecr_pull_through_rule(self) -> None:
        logging.info(
            f"Checking existing pull through cache rules for '{self.ecr_namespace}'",
        )

        try:
            self.ecr_client.describe_pull_through_cache_rules(
                ecrRepositoryPrefixes=[self.ecr_namespace],
            )
        except self.ecr_client.exceptions._code_to_exception[
            "PullThroughCacheRuleNotFoundException"
        ]:
            logging.info(
                f"""Pull through cache rule for '{self.ecr_namespace}' not found..
                creating pull through cache rule for '{self.ecr_namespace}'""",
            )

            self.ecr_client.create_pull_through_cache_rule(
                ecrRepositoryPrefix=self.ecr_namespace,
                upstreamRegistryUrl="public.ecr.aws",
            )

            logging.info(
                f"Pull through cache rule for '{self.ecr_namespace}' created successfully",
            )

    def _patch_uri(self, og_uri: str) -> str:
        full_ref, tag = og_uri.split(":")

        ref, *repo = full_ref.split("/")

        def pull_through(repository_name):
            _ = self.ecr_client.batch_get_image(
                repositoryName=repository_name,
                imageIds=[{"imageTag": tag}],
            )
            repo_response = self.ecr_client.describe_repositories(
                repositoryNames=[repository_name],
            )
            return repo_response["repositories"][0]["repositoryUri"] + ":" + tag

        if ref == "fndnt":
            logging.info("Reusable component detected, patching URI")
            uri = pull_through(f"{self.ecr_namespace}/{full_ref}")
        elif ref == "public.ecr.aws":
            logging.info("Public AWS ECR component detected, patching URI")
            uri = pull_through(f"{self.ecr_namespace}/{'/'.join(repo)}")
        else:
            logging.info("Custom component detected")
            # the uri does not need patching
            uri = og_uri
        return uri

    def validate_base_path(self, base_path: str) -> None:
        if not base_path.startswith("s3://"):
            msg = "base_path must be a valid s3 path, starting with s3://"
            raise ValueError(msg)

        if base_path.endswith("/"):
            msg = "base_path must not end with a '/'"
            raise ValueError(msg)

    def compile(
        self,
        dataset: Dataset,
        working_directory: str,
        output_path: str,
        *,
        role_arn: t.Optional[str] = None,
    ) -> None:
        """Compile a fondant dataset workflow to sagemaker pipeline spec and save it
        to a specified output path.

        Args:
            dataset: the dataset to compile
            working_directory: path of the working directory
            output_path: the path where to save the sagemaker pipeline spec.
            role_arn: the Amazon Resource Name role to use for the processing steps,
            if none provided the `sagemaker.get_execution_role()` role will be used.
        """
        self.ecr_client = self.boto3.client("ecr")
        self.validate_base_path(working_directory)
        self._check_ecr_pull_through_rule()

        run_id = dataset.manifest.run_id
        dataset.validate()

        component_cache_key = None

        steps: t.List[t.Any] = []

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdirname:
            for component_name, component in dataset._graph.items():
                component_op = component["operation"]
                component_cache_key = component_op.get_component_cache_key(
                    previous_component_cache=component_cache_key,
                )

                metadata = Metadata(
                    run_id=run_id,
                    component_id=component_name,
                    cache_key=component_cache_key,
                    dataset_name=dataset.name,
                    manifest_location=f"{working_directory}/{dataset.name}/{run_id}/"
                    f"{component_name}/manifest.json",
                )

                logger.info(f"Compiling service for {component_name}")

                command = self._build_command(
                    metadata=metadata,
                    working_directory=working_directory,
                    arguments=component_op.arguments,
                    dependencies=component["dependencies"],
                )
                depends_on = [steps[-1]] if component["dependencies"] else []

                image = component_op.image
                entrypoint = self._build_entrypoint(image)

                script_path = self.generate_component_script(
                    entrypoint=entrypoint,
                    command=command,
                    component_name=component_name,
                    directory=tmpdirname,
                )

                if not role_arn:
                    # if no role is provided use the default sagemaker execution role
                    # https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-ex-role.html
                    role_arn = self.sagemaker.get_execution_role()

                resources_dict = self._set_configuration(component_op)

                processor = self.sagemaker.processing.ScriptProcessor(
                    image_uri=self._patch_uri(image.base_image),
                    command=["bash"],
                    instance_count=1,
                    base_job_name=component_name,
                    role=role_arn,
                    **resources_dict,
                )

                step = self.sagemaker.workflow.steps.ProcessingStep(
                    name=component_name,
                    processor=processor,
                    depends_on=depends_on,
                    code=script_path,
                )

                steps.append(step)

            sagemaker_pipeline = self.sagemaker.workflow.pipeline.Pipeline(
                name=dataset.name,
                steps=steps,
            )
            with open(output_path, "w") as outfile:
                json.dump(
                    json.loads(sagemaker_pipeline.definition()),
                    outfile,
                    indent=4,
                )

    def _set_configuration(
        self,
        fondant_component_operation,
        *args,
        **kwargs,
    ):
        # Used configurations
        resources_dict = fondant_component_operation.resources.to_dict()

        instance_type = resources_dict.pop("instance_type")

        if not instance_type:
            logger.warning(
                """No instance type provided, using default `ml.t3.medium`. See:
            https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html
            for options""",
            )
            instance_type = "ml.t3.medium"

        # Unused configurations
        self.log_unused_configurations(**resources_dict)

        return {"instance_type": instance_type}

    @staticmethod
    def generate_component_script(
        *,
        entrypoint: t.List[str],
        command: t.List[str],
        component_name: str,
        directory: str,
    ) -> str:
        """Generate a bash script for a component to be used as input in a
        sagemaker pipeline step. Returns the path to the script.
        """
        # use shlex.quote to escape special bash chars
        command_string = [arg.replace("'", "") for arg in command]
        cleaned_script = shlex.join([*entrypoint, *command_string])

        with open(f"{directory}/{component_name}.sh", "w") as f:
            f.write(cleaned_script)
        return f"{directory}/{component_name}.sh"
