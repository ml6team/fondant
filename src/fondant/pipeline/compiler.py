import json
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from fondant.core.exceptions import InvalidPipelineDefinition
from fondant.core.manifest import Metadata
from fondant.pipeline import (
    VALID_ACCELERATOR_TYPES,
    VALID_VERTEX_ACCELERATOR_TYPES,
    Pipeline,
)

logger = logging.getLogger(__name__)

DASK_DIAGNOSTIC_DASHBOARD_PORT = 8787
KubeflowCommandArguments = t.List[t.Union[str, t.Dict[str, str]]]


@dataclass
class Accelerator:
    """
    Represents a hardware accelerator configuration.

    Args:
        type: Type of the accelerator.
        number: The number of the accelerator.
    """

    type: str
    number: int


@dataclass
class ComponentConfigs:
    """
    Represents the configurations for a component.

    Args:
        image: The Docker image for the component.
        arguments: Arguments to be passed to the component.
        dependencies: List of dependencies required for the component.
        accelerators: List of hardware accelerators for the component.
        cpu_request: CPU request for the component.
        cpu_limit: CPU limit for the component.
        memory_request: Memory request for the component.
        memory_limit: Memory limit for the component.
    """

    image: t.Optional[str] = None
    arguments: t.Optional[t.Dict[str, t.Any]] = None
    dependencies: t.Optional[t.List[str]] = None
    accelerators: t.Optional[t.List[Accelerator]] = None
    cpu_request: t.Optional[str] = None
    cpu_limit: t.Optional[str] = None
    memory_request: t.Optional[str] = None
    memory_limit: t.Optional[str] = None


@dataclass
class KubeflowComponentConfig(ComponentConfigs):
    """
    Represents Kubeflow-specific configurations for a component.

    Args:
        node_pool_label: Label for the node pool.
        node_pool_name: Name of the node pool.
    """

    node_pool_label: t.Optional[str] = None
    node_pool_name: t.Optional[str] = None


@dataclass
class DockerComponentConfig(ComponentConfigs):
    """
    Represents Docker-specific configurations for a component.

    Args:
        context: The context for the Docker component.
        volumes: List of volumes for the Docker component.
        ports: List of ports for the Docker component.
    """

    context: t.Optional[str] = None
    volumes: t.Optional[t.List[t.Union[str, dict]]] = None
    ports: t.Optional[t.List[t.Union[str, dict]]] = None


@dataclass
class PipelineConfigs:
    """
    Represents the configurations for a pipeline.

    Args:
        pipeline_name: Name of the pipeline.
        pipeline_description: Description of the pipeline.
    """

    pipeline_name: str
    pipeline_description: str
    pipeline_version: str


@dataclass
class DockerPipelineConfigs(PipelineConfigs):
    """
    Represents Docker-specific configurations for a pipeline.

    Args:
       component_configs: Dictionary of Docker component configurations for the pipeline.
    """

    component_configs: t.Dict[str, DockerComponentConfig]


@dataclass
class KubeflowPipelineConfigs(PipelineConfigs):
    """
    Represents Kubeflow-specific configurations for a pipeline.

    Args:
        component_configs: Dictionary of Kubeflow component configurations for the pipeline.
    """

    component_configs: t.Dict[str, KubeflowComponentConfig]


class Compiler(ABC):
    """Abstract base class for a compiler."""

    @abstractmethod
    def compile(self, *args, **kwargs) -> None:
        """Abstract method to invoke compilation."""

    @abstractmethod
    def _set_configuration(self, *args, **kwargs) -> None:
        """Abstract method to set pipeline configuration."""

    @staticmethod
    @abstractmethod
    def get_pipeline_configs(path: str) -> PipelineConfigs:
        """
        Abstract method to get pipeline configs from a pipeline specification.

        Args:
            path: path to the pipeline specification

        Returns:
            PipelineConfigs object containing the pipeline configs
        """


@dataclass
class DockerVolume:
    """Dataclass representing a DockerVolume.
    (https://docs.docker.com/compose/compose-file/05-services/#volumes).

    Args:
        type: the mount type volume (bind, volume)
        source: the source of the mount, a path on the host for a bind mount
        target: the path in the container where the volume is mounted.
    """

    type: str
    source: str
    target: str


class DockerCompiler(Compiler):
    """Compiler that creates a docker-compose spec from a pipeline."""

    def compile(
        self,
        pipeline: Pipeline,
        *,
        output_path: str = "docker-compose.yml",
        extra_volumes: t.Optional[list] = None,
        build_args: t.Optional[t.List[str]] = None,
    ) -> None:
        """Compile a pipeline to docker-compose spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the docker-compose spec
            extra_volumes: a list of extra volumes (using the Short syntax:
              https://docs.docker.com/compose/compose-file/05-services/#short-syntax-5)
              to mount in the docker-compose spec.
            build_args: List of build arguments to pass to docker
        """
        if extra_volumes is None:
            extra_volumes = []

        logger.info(f"Compiling {pipeline.name} to {output_path}")
        spec = self._generate_spec(
            pipeline,
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
        p_base_path = Path(base_path)
        # check if base path is an existing local folder
        if p_base_path.exists():
            logger.info(
                f"Base path found on local system, setting up {base_path} as mount volume",
            )
            p_base_path = p_base_path.resolve()
            volume = DockerVolume(
                type="bind",
                source=str(p_base_path),
                target=f"/{p_base_path.stem}",
            )
            path = f"/{p_base_path.stem}"
        else:
            logger.info(f"Base path {base_path} is remote")
            volume = None
            path = base_path
        return path, volume

    def _generate_spec(
        self,
        pipeline: Pipeline,
        *,
        extra_volumes: t.List[str],
        build_args: t.List[str],
    ) -> dict:
        """Generate a docker-compose spec as a python dictionary,
        loops over the pipeline graph to create services and their dependencies.
        """
        path, volume = self._patch_path(base_path=pipeline.base_path)
        run_id = pipeline.get_run_id()

        services = {}

        pipeline.validate(run_id=run_id)

        component_cache_key = None

        for component_name, component in pipeline._graph.items():
            component_op = component["fondant_component_op"]

            component_cache_key = component_op.get_component_cache_key(
                previous_component_cache=component_cache_key,
            )

            metadata = Metadata(
                pipeline_name=pipeline.name,
                run_id=run_id,
                base_path=path,
                component_id=component_name,
                cache_key=component_cache_key,
            )

            logger.info(f"Compiling service for {component_name}")

            # add metadata argument to command
            command = ["--metadata", metadata.to_json()]

            # add in and out manifest paths to command
            command.extend(
                [
                    "--output_manifest_path",
                    f"{path}/{metadata.pipeline_name}/{metadata.run_id}/"
                    f"{component_name}/manifest.json",
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
                    # there is only an input manifest if the component has dependencies
                    command.extend(
                        [
                            "--input_manifest_path",
                            f"{path}/{metadata.pipeline_name}/{metadata.run_id}/"
                            f"{dependency}/manifest.json",
                        ],
                    )

            volumes: t.List[t.Union[str, dict]] = []
            if volume:
                volumes.append(asdict(volume))
            if extra_volumes:
                volumes.extend(extra_volumes)

            ports: t.List[t.Union[str, dict]] = []
            ports.append(
                f"{DASK_DIAGNOSTIC_DASHBOARD_PORT}:{DASK_DIAGNOSTIC_DASHBOARD_PORT}",
            )

            services[component_name] = {
                "command": command,
                "depends_on": depends_on,
                "volumes": volumes,
                "ports": ports,
            }

            self._set_configuration(services, component_op, component_name)

            if component_op.dockerfile_path is not None:
                logger.info(
                    f"Found Dockerfile for {component_name}, adding build step.",
                )
                services[component_name]["build"] = {
                    "context": str(component_op.component_dir),
                    "args": build_args,
                }
            else:
                services[component_name]["image"] = component_op.component_spec.image

        return {
            "name": pipeline.name,
            "version": "3.8",
            "services": services,
            "labels": {
                "description": pipeline.description,
            },
        }

    @staticmethod
    def _set_configuration(services, fondant_component_operation, component_name):
        accelerator_name = fondant_component_operation.accelerator_name
        accelerator_number = fondant_component_operation.number_of_accelerators

        if accelerator_name is not None:
            if accelerator_name not in VALID_ACCELERATOR_TYPES:
                msg = (
                    f"Configured accelerator `{accelerator_name}`"
                    f" is not a valid accelerator type for Docker Compose compiler."
                    f" Available options: {VALID_VERTEX_ACCELERATOR_TYPES}"
                )
                raise InvalidPipelineDefinition(msg)

            if accelerator_name == "GPU":
                services[component_name]["deploy"] = {
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

    @staticmethod
    def get_pipeline_configs(path: str) -> DockerPipelineConfigs:
        """Get pipeline configs from a pipeline specification."""
        with open(path) as file_:
            specification = yaml.safe_load(file_)

        components_configs_dict = {}

        # Iterate through each service
        for component_name, component_configs in specification["services"].items():
            # Get arguments from command
            command_list = component_configs.get("command", [])
            arguments = {}
            for i in range(0, len(command_list), 2):
                arguments[command_list[i].lstrip("-")] = command_list[i + 1]

            # Get accelerator name and number of accelerators
            resources = component_configs.get("deploy", {}).get("resources", {})
            devices = resources.get("reservations", {}).get("devices", {})

            accelerator_list = []
            if devices:
                for device in devices:
                    accelerator = Accelerator(
                        type=device["capabilities"][0],
                        number=device["count"],
                    )
                    accelerator_list.append(accelerator)

            component_config = DockerComponentConfig(
                image=component_configs.get("image", None),
                arguments=arguments,
                dependencies=list(component_configs.get("depends_on", {}).keys()),
                accelerators=accelerator_list,
                context=component_configs.get("build", {}).get("context", None),
                ports=component_configs.get("ports", None),
                volumes=component_configs.get("volumes", None),
                cpu_request=None,
                cpu_limit=None,
                memory_request=None,
                memory_limit=None,
            )
            components_configs_dict[component_name] = component_config

        return DockerPipelineConfigs(
            pipeline_name=specification["name"],
            pipeline_version=specification["version"],
            pipeline_description=specification.get("labels", {}).get(
                "description",
                None,
            ),
            component_configs=components_configs_dict,
        )


class KubeFlowCompiler(Compiler):
    """Compiler that creates a Kubeflow pipeline spec from a pipeline."""

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
        pipeline: Pipeline,
        output_path: str,
    ) -> None:
        """Compile a pipeline to Kubeflow pipeline spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the Kubeflow pipeline spec
        """
        run_id = pipeline.get_run_id()
        pipeline.validate(run_id=run_id)
        logger.info(f"Compiling {pipeline.name} to {output_path}")

        def set_component_exec_args(
            *,
            component_op,
            component_args: t.List[str],
            input_manifest_path: bool,
        ):
            """Dump Fondant specification arguments to kfp command executor arguments."""
            dumped_args: KubeflowCommandArguments = []

            component_args.extend(["output_manifest_path", "metadata"])
            if input_manifest_path:
                component_args.append("input_manifest_path")

            for arg in component_args:
                arg_name = arg.strip().replace(" ", "_")
                arg_name_cmd = f"--{arg_name}"

                dumped_args.append(arg_name_cmd)
                dumped_args.append("{{$.inputs.parameters['" + f"{arg_name}" + "']}}")

            component_op.component_spec.implementation.container.args = dumped_args

            return component_op

        @self.kfp.dsl.pipeline(name=pipeline.name, description=pipeline.description)
        def kfp_pipeline():
            previous_component_task = None
            component_cache_key = None

            for component_name, component in pipeline._graph.items():
                logger.info(f"Compiling service for {component_name}")

                component_op = component["fondant_component_op"]
                # convert ComponentOp to Kubeflow component
                kubeflow_component_op = self.kfp.components.load_component_from_text(
                    text=component_op.component_spec.kubeflow_specification.to_string(),
                )

                # Remove None values from arguments
                component_args = {
                    k: v for k, v in component_op.arguments.items() if v is not None
                }

                component_cache_key = component_op.get_component_cache_key(
                    previous_component_cache=component_cache_key,
                )
                metadata = Metadata(
                    pipeline_name=pipeline.name,
                    run_id=run_id,
                    base_path=pipeline.base_path,
                    component_id=component_name,
                    cache_key=component_cache_key,
                )

                output_manifest_path = (
                    f"{pipeline.base_path}/{pipeline.name}/"
                    f"{run_id}/{component_name}/manifest.json"
                )
                # Set the execution order of the component task to be after the previous
                # component task.
                if component["dependencies"]:
                    for dependency in component["dependencies"]:
                        input_manifest_path = (
                            f"{pipeline.base_path}/{pipeline.name}/"
                            f"{run_id}/{dependency}/manifest.json"
                        )
                        kubeflow_component_op = set_component_exec_args(
                            component_op=kubeflow_component_op,
                            component_args=list(component_args.keys()),
                            input_manifest_path=True,
                        )
                        component_task = kubeflow_component_op(
                            input_manifest_path=input_manifest_path,
                            output_manifest_path=output_manifest_path,
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

        logger.info(f"Compiling {pipeline.name} to {output_path}")

        self.kfp.compiler.Compiler().compile(kfp_pipeline, output_path)  # type: ignore
        logger.info("Pipeline compiled successfully")

    def _set_configuration(self, task, fondant_component_operation):
        # Unpack optional specifications
        number_of_accelerators = fondant_component_operation.number_of_accelerators
        accelerator_name = fondant_component_operation.accelerator_name
        node_pool_label = fondant_component_operation.node_pool_label
        node_pool_name = fondant_component_operation.node_pool_name
        cpu_request = fondant_component_operation.cpu_request
        cpu_limit = fondant_component_operation.cpu_limit
        memory_request = fondant_component_operation.memory_request
        memory_limit = fondant_component_operation.memory_limit

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
                raise InvalidPipelineDefinition(msg)

            task.set_accelerator_limit(number_of_accelerators)
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

    @staticmethod
    def get_pipeline_configs(path: str) -> KubeflowPipelineConfigs:
        """Get pipeline configs from a pipeline specification."""
        # Two specs are present and loaded in the yaml file (component spec and k8s specs)
        k8_specification = {}
        specification = {}

        with open(path) as file_:
            for spec in yaml.load_all(file_, Loader=yaml.FullLoader):
                if "deploymentSpec" in spec:
                    specification = spec
                elif "platforms" in spec:
                    k8_specification = spec["platforms"]["kubernetes"][
                        "deploymentSpec"
                    ]["executors"]

        if not specification:
            msg = "No component specification found in the pipeline specification"
            raise InvalidPipelineDefinition(msg)
        components_configs_dict = {}

        # Iterate through each service
        for component_name, component_configs in specification["root"]["dag"][
            "tasks"
        ].items():
            # Get arguments from command
            arguments = {
                arg_name: arg_value["runtimeValue"]["constant"]
                for arg_name, arg_value in component_configs["inputs"][
                    "parameters"
                ].items()
            }

            # Get accelerator name and number of accelerators
            container_spec = specification["deploymentSpec"]["executors"][
                f"exec-{component_name}"
            ]["container"]
            resources = component_configs.get("resources", {})
            devices = resources.get("accelerator", {})
            accelerator_list = []

            if devices:
                for device in devices:
                    accelerator = Accelerator(
                        type=device["accelerator"]["type"],
                        number=device["accelerator"]["count"],
                    )
                    accelerator_list.append(accelerator)

            # Get node pool name and label
            node_pool_label = None
            node_pool_name = None
            if k8_specification:
                node_pool_dict = (
                    k8_specification.get(f"exec-{component_name}", {})
                    .get("nodeSelector", {})
                    .get("labels", {})
                )
                if node_pool_dict:
                    node_pool_label = list(node_pool_dict.keys())[0]
                    node_pool_name = list(node_pool_dict.values())[0]

            component_config = KubeflowComponentConfig(
                image=container_spec.get("image"),
                arguments=arguments,
                dependencies=component_configs.get("dependentTasks", []),
                accelerators=accelerator_list,
                cpu_request=component_configs.get("cpuRequest", None),
                cpu_limit=component_configs.get("cpuLimit", None),
                memory_request=component_configs.get("memoryRequest", None),
                memory_limit=component_configs.get("memoryLimit", None),
                node_pool_name=node_pool_name,
                node_pool_label=node_pool_label,
            )
            components_configs_dict[component_name] = component_config

        pipeline_info = specification["pipelineInfo"]

        return KubeflowPipelineConfigs(
            pipeline_name=pipeline_info["name"],
            pipeline_version=specification["sdkVersion"],
            pipeline_description=pipeline_info.get("description", None),
            component_configs=components_configs_dict,
        )


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
        # Unpack optional specifications
        cpu_limit = fondant_component_operation.cpu_limit
        memory_limit = fondant_component_operation.memory_limit
        number_of_accelerators = fondant_component_operation.number_of_accelerators
        accelerator_name = fondant_component_operation.accelerator_name

        # Assign optional specification
        if cpu_limit is not None:
            task.set_cpu_limit(cpu_limit)
        if memory_limit is not None:
            task.set_memory_limit(memory_limit)
        if number_of_accelerators is not None:
            task.set_accelerator_limit(number_of_accelerators)
            if accelerator_name not in VALID_VERTEX_ACCELERATOR_TYPES:
                msg = (
                    f"Configured accelerator `{accelerator_name}` is not a valid accelerator type"
                    f"for Vertex compiler. Available options: {VALID_VERTEX_ACCELERATOR_TYPES}"
                )
                raise InvalidPipelineDefinition(msg)

            task.set_accelerator_type(accelerator_name)

        return task
