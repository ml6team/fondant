import json
import typing as t
from abc import abstractmethod
from dataclasses import dataclass

import yaml

from fondant.core.exceptions import InvalidDatasetDefinition


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
    command: t.Optional[t.List[str]] = None
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
class DatasetConfigs:
    """
    Represents the configurations for a dataset workflow.

    Args:
        dataset_name: Name of the dataset.
        dataset_description: Description of the dataset.
    """

    dataset_name: str
    dataset_version: str
    dataset_description: t.Optional[str] = None

    @classmethod
    @abstractmethod
    def from_spec(cls, spec_path: str) -> "DatasetConfigs":
        """Get dataset configs from a dataset workflow specification."""


@dataclass
class DockerComposeConfigs(DatasetConfigs):
    """
    Represents Docker-specific configurations for a dataset.

    Args:
       component_configs: Dictionary of Docker component configurations for the dataset.
    """

    component_configs: t.Optional[t.Dict[str, DockerComponentConfig]] = None

    @classmethod
    def from_spec(cls, spec_path: str) -> "DockerComposeConfigs":
        """Get dataset configs from a dataset workflow specification."""
        with open(spec_path) as file_:
            specification = yaml.safe_load(file_)

        components_configs_dict = {}

        dataset_description = None
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
            dataset_description = component_configs.get("labels", {}).get(
                "dataset_description",
                "No description provided",
            )

        return cls(
            dataset_name=specification["name"],
            dataset_version=specification["version"],
            dataset_description=dataset_description,
            component_configs=components_configs_dict,
        )


@dataclass
class KubeflowPipelineConfigs(DatasetConfigs):
    """
    Represents Kubeflow-specific configurations for a dataset.

    Args:
        component_configs: Dictionary of Kubeflow component configurations for the dataset.
    """

    component_configs: t.Optional[t.Dict[str, KubeflowComponentConfig]] = None

    @classmethod
    def from_spec(cls, spec_path: str) -> "KubeflowPipelineConfigs":
        """Get dataset configs from a dataset specification."""
        # Two specs are present and loaded in the yaml file (component spec and k8s specs)
        k8_specification = {}
        specification = {}

        with open(spec_path) as file_:
            for spec in yaml.load_all(file_, Loader=yaml.FullLoader):
                if "deploymentSpec" in spec:
                    specification = spec
                elif "platforms" in spec:
                    k8_specification = spec["platforms"]["kubernetes"][
                        "deploymentSpec"
                    ]["executors"]

        if not specification:
            msg = "No component specification found in the dataset specification"
            raise InvalidDatasetDefinition(msg)
        components_configs_dict = {}

        # Iterate through each service
        for sanitized_component_name, component_configs in specification["root"]["dag"][
            "tasks"
        ].items():
            # Get arguments from command
            arguments = {
                arg_name: arg_value["runtimeValue"]["constant"]
                for arg_name, arg_value in component_configs["inputs"][
                    "parameters"
                ].items()
            }

            component_name = json.loads(arguments["metadata"])["component_id"]
            # Get accelerator name and number of accelerators
            container_spec = specification["deploymentSpec"]["executors"][
                f"exec-{sanitized_component_name}"
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
                    k8_specification.get(f"exec-{sanitized_component_name}", {})
                    .get("nodeSelector", {})
                    .get("labels", {})
                )
                if node_pool_dict:
                    node_pool_label = list(node_pool_dict.keys())[0]
                    node_pool_name = list(node_pool_dict.values())[0]

            dependencies = component_configs.get("dependentTasks", [])

            if dependencies:
                dependencies = [
                    dependency.replace("-", "_") for dependency in dependencies
                ]

            component_config = KubeflowComponentConfig(
                image=container_spec.get("image"),
                command=container_spec.get("command"),
                arguments=arguments,
                dependencies=dependencies,
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

        return cls(
            dataset_name=pipeline_info["name"],
            dataset_version=specification["sdkVersion"],
            dataset_description=pipeline_info.get("description", None),
            component_configs=components_configs_dict,
        )


@dataclass
class VertexPipelineConfigs(KubeflowPipelineConfigs):
    pass
