"""This module defines classes to represent a Fondant Pipeline."""
import hashlib
import json
import logging
import re
import subprocess  # nosec
import typing as t
from collections import OrderedDict
from pathlib import Path

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from fondant.component_spec import ComponentSpec
from fondant.exceptions import InvalidImageDigest, InvalidPipelineDefinition
from fondant.filesystem import list_files
from fondant.manifest import Manifest
from fondant.schema import validate_partition_number

logger = logging.getLogger(__name__)


class ComponentOp:
    """
    Class representing an operation for a Fondant Component in a Kubeflow Pipeline. An operation
    is a representation of a function that will be executed as part of a pipeline.

    Arguments:
        component_dir: The path to the component directory.
        arguments: A dictionary containing the argument name and value for the operation.
        input_partition_rows: The number of rows to load per partition. Set to override the
        automatic partitioning
        number_of_gpus: The number of gpus to assign to the operation
        node_pool_label: The label of the node pool to which the operation will be assigned.
        node_pool_name: The name of the node pool to which the operation will be assigned.

    Note:
        - A Fondant Component operation is created by defining a Fondant Component and its input
          arguments.
        - The `number_of_gpus`, `node_pool_label`, `node_pool_name`
         attributes are optional and can be used to specify additional
          configurations for the operation. More information on the optional attributes that can
          be assigned to kfp components here:
          https://kubeflow-pipelines.readthedocs.io/en/1.8.13/source/kfp.dsl.html
    """

    COMPONENT_SPEC_NAME = "fondant_component.yaml"

    def __init__(
        self,
        component_dir: t.Union[str, Path],
        *,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[str, int]] = None,
        number_of_gpus: t.Optional[int] = None,
        node_pool_label: t.Optional[str] = None,
        node_pool_name: t.Optional[str] = None,
    ) -> None:
        self.component_dir = Path(component_dir)
        self.component_spec = ComponentSpec.from_file(
            self.component_dir / self.COMPONENT_SPEC_NAME,
        )
        self.name = self.component_spec.name.replace(" ", "_").lower()
        self.input_partition_rows = input_partition_rows
        self.arguments = self._set_arguments(arguments)
        self.arguments.setdefault("component_spec", self.component_spec.specification)
        self.number_of_gpus = number_of_gpus
        self.node_pool_label, self.node_pool_name = self._validate_node_pool_spec(
            node_pool_label,
            node_pool_name,
        )

    def _set_arguments(
        self,
        arguments: t.Optional[t.Dict[str, t.Any]],
    ) -> t.Dict[str, t.Any]:
        """Set component arguments based on provided arguments and relevant ComponentOp
        parameters.
        """
        arguments = arguments or {}

        input_partition_rows = validate_partition_number(self.input_partition_rows)

        arguments["input_partition_rows"] = str(input_partition_rows)

        return arguments

    def _validate_node_pool_spec(
        self,
        node_pool_label,
        node_pool_name,
    ) -> t.Tuple[t.Optional[str], t.Optional[str]]:
        """Validate node pool specification."""
        if bool(node_pool_label) != bool(node_pool_name):
            msg = "Both node_pool_label and node_pool_name must be specified or both must be None."
            raise InvalidPipelineDefinition(
                msg,
            )
        return node_pool_label, node_pool_name

    @property
    def dockerfile_path(self) -> t.Optional[Path]:
        path = self.component_dir / "Dockerfile"
        return path if path.exists() else None

    @classmethod
    def from_registry(
        cls,
        name: str,
        *,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        number_of_gpus: t.Optional[int] = None,
        node_pool_label: t.Optional[str] = None,
        node_pool_name: t.Optional[str] = None,
    ) -> "ComponentOp":
        """Load a reusable component by its name.

        Args:
            name: Name of the component to load
            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            number_of_gpus: The number of gpus to assign to the operation
            node_pool_label: The label of the node pool to which the operation will be assigned.
            node_pool_name: The name of the node pool to which the operation will be assigned.
        """
        components_dir: Path = t.cast(Path, files("fondant") / f"components/{name}")

        if not (components_dir.exists() and components_dir.is_dir()):
            msg = f"No reusable component with name {name} found."
            raise ValueError(msg)

        return ComponentOp(
            components_dir,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            number_of_gpus=number_of_gpus,
            node_pool_label=node_pool_label,
            node_pool_name=node_pool_name,
        )

    @staticmethod
    def get_image_manifest(image_ref: str):
        """Retrieve the Docker image manifest.

        Args:
            image_ref: The Docker image reference (e.g., 'registry/image:tag').

        Returns:
            dict: The parsed JSON manifest.
        """
        cmd = ["docker", "manifest", "inspect", image_ref]
        try:
            result = subprocess.run(  # nosec
                cmd,
                capture_output=True,
                check=True,
                text=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            msg = f"Error executing command: {e}"
            raise Exception(msg)
        except json.JSONDecodeError as e:
            msg = f"Error decoding response: {e}"
            raise Exception(msg)

    def get_component_image_hash(self, image_ref):
        """Calculate the hash of a Docker image reference. If multiple builds exist for
        different operating systems, the hash of the amd64 CPU architecture and the linux
        operating system is returned.

        Args:
            image_ref (str): The Docker image reference (e.g., 'registry/image:tag').

        Returns:
            str: The hash value (digest) of the Docker image.
        """
        manifest = self.get_image_manifest(image_ref)

        hash_ = None

        response_manifests = manifest.get("manifests", [])
        for response_manifest in response_manifests:
            platform_specs = response_manifest.get("platform", {})
            if (
                platform_specs.get("architecture") == "amd64"
                and platform_specs.get("os") == "linux"
            ):
                hash_ = response_manifest.get("digest")
                break

        if not hash_:
            config_digest = manifest.get("config", {}).get("digest")
            if not config_digest:
                msg = "No valid digest found in the image manifest."
                raise InvalidImageDigest(msg)
            hash_ = config_digest

        return hash_

    def get_component_cache_key(self) -> str:
        """Calculate a cache key representing the unique identity of this ComponentOp.

        The cache key is computed based on the component specification, image hash, arguments, and
        other attributes of the ComponentOp. It is used to uniquely identify a specific instance
        of the ComponentOp and is used for caching.

        Returns:
            A cache key representing the unique identity of this ComponentOp.
        """

        def sorted_dict_to_json(input_dict):
            """Convert a dictionary to a sorted JSON string.

            This function recursively converts nested dictionaries to ensure all dictionaries
            are sorted and their values are JSON-compatible (e.g., lists, dictionaries, strings,
            numbers, booleans, or None).

            Args:
            input_dict: The dictionary to be converted.

            Returns:
            A sorted JSON string representing the dictionary.
            """
            if isinstance(input_dict, dict):
                return json.dumps(
                    {k: sorted_dict_to_json(v) for k, v in sorted(input_dict.items())},
                )

            return input_dict

        def get_nested_dict_hash(input_dict):
            """Calculate the hash of a nested dictionary.

            Args:
                input_dict: The nested dictionary to calculate the hash for.

            Returns:
                The hash value (MD5 digest) of the nested dictionary.
            """
            sorted_json_string = sorted_dict_to_json(input_dict)
            hash_object = hashlib.md5(sorted_json_string.encode())  # nosec
            return hash_object.hexdigest()

        component_spec_dict = self.component_spec.specification

        component_op_uid_dict = {
            "component_spec_hash": get_nested_dict_hash(component_spec_dict),
            "component_image_hash": self.get_component_image_hash(
                component_spec_dict["image"],
            ),
            "arguments": get_nested_dict_hash(self.arguments)
            if self.arguments is not None
            else None,
            "input_partition_rows": self.input_partition_rows,
            "number_of_gpus": self.number_of_gpus,
            "node_pool_name": self.node_pool_name,
        }

        return get_nested_dict_hash(component_op_uid_dict)


class Pipeline:
    """Class representing a Fondant Pipeline."""

    def __init__(
        self,
        base_path: str,
        pipeline_name: str,
        pipeline_description: t.Optional[str] = None,
    ):
        """
        Args:
            base_path: The base path for the pipeline where the artifacts are stored.
            pipeline_name: The name of the pipeline.
            pipeline_description: Optional description of the pipeline.
        """
        self.base_path = base_path
        self.name = self._validate_pipeline_name(pipeline_name)
        self.description = pipeline_description
        self.package_path = f"{pipeline_name}.tgz"
        self._graph: t.OrderedDict[str, t.Any] = OrderedDict()
        self.task_without_dependencies_added = False

    def add_op(
        self,
        task: ComponentOp,
        dependencies: t.Optional[t.Union[ComponentOp, t.List[ComponentOp]]] = None,
    ):
        """
        Add a task to the pipeline with an optional dependency.

        Args:
            task: The task to add to the pipeline.
            dependencies: Optional task dependencies that needs to be completed before the task
             can run.
        """
        if dependencies is None:
            if self.task_without_dependencies_added:
                msg = "At most one task can be defined without dependencies."
                raise InvalidPipelineDefinition(
                    msg,
                )
            dependencies = []
            self.task_without_dependencies_added = True
        elif not isinstance(dependencies, list):
            dependencies = [dependencies]

        if len(dependencies) > 1:
            msg = (
                f"Multiple component dependencies provided for component "
                f"`{task.component_spec.name}`. The current version of Fondant can only handle "
                f"components with a single dependency. Please note that the behavior of the "
                f"pipeline may be unpredictable or incorrect."
            )
            raise InvalidPipelineDefinition(
                msg,
            )

        dependencies_names = [dependency.name for dependency in dependencies]

        self._graph[task.name] = {
            "fondant_component_op": task,
            "dependencies": dependencies_names,
        }

    def sort_graph(self):
        """Sort the graph topologically based on task dependencies."""
        logger.info("Sorting pipeline component graph topologically.")
        sorted_graph = []
        visited = set()

        def depth_first_traversal(node: str):
            """
            Perform a depth-first traversal of the graph and its dependencies.

            Args:
                node: The name of the starting node for traversal.
            """
            if node not in visited:
                visited.add(node)
                for dependency in self._graph[node]["dependencies"]:
                    depth_first_traversal(dependency)
                sorted_graph.append(node)

        for graph_node in self._graph:
            depth_first_traversal(graph_node)

        self._graph = OrderedDict((node, self._graph[node]) for node in sorted_graph)

    @staticmethod
    def _validate_pipeline_name(pipeline_name: str) -> str:
        pattern = r"^[a-z0-9][a-z0-9_-]*$"
        if not re.match(pattern, pipeline_name):
            msg = f"The pipeline name violates the pattern {pattern}"
            raise InvalidPipelineDefinition(msg)
        return pipeline_name

    def validate(self, run_id: str):
        """Sort and run validation on the pipeline definition.

        Args:
            run_id (str, optional): run identifier. Defaults to None.
        """
        self.sort_graph()
        self._validate_pipeline_definition(run_id)

    def execute_component(self, component_name: str, cache_key: str) -> bool:
        """
        Function that checks whether a component should be executed
        Args:
            component_name: the name of the component to execute
            cache_key: the component cache key
        Return:
            boolean indicating whether to execute the component.
        """
        executed_manifest_dir = f"{self.base_path}/{component_name}"
        executed_manifests = [
            file.rsplit("/", 1)[-1] for file in list_files(executed_manifest_dir)
        ]
        current_manifest = f"manifest_{cache_key}.json"
        manifest_path = f"{executed_manifest_dir}/{current_manifest}"

        if current_manifest in executed_manifests:
            manifest = Manifest.from_file(manifest_path)
            logger.info(
                f"Found matching execution for component '{component_name}' under"
                f" {manifest_path} with run_id {manifest.run_id}\n."
                f" Caching component execution.",
            )
            return False

        return True

    def _validate_pipeline_definition(self, run_id: str):
        """
        Validates the pipeline definition by ensuring that the consumed and produced subsets and
        their associated fields match and are invoked in the correct order.

        Raises:
            InvalidPipelineDefinition: If a component is trying to invoke a subset that is not
             defined or created in previous components, or if an invoked subset's schema does not
              match the previously created subset definition.
            base_path: the base path where to store the pipelines artifacts
            run_id: the run id of the component
        """
        if len(self._graph.keys()) == 0:
            logger.info("No components defined in the pipeline. Nothing to validate.")
            return

        # TODO: change later if we decide to run 2 fondant pipelines after each other
        load_component = True
        load_component_name = list(self._graph.keys())[0]

        # Create initial manifest
        manifest = Manifest.create(
            base_path=self.base_path,
            run_id=run_id,
            component_id=load_component_name,
            cache_key="42",
        )
        for operation_specs in self._graph.values():
            fondant_component_op = operation_specs["fondant_component_op"]
            component_spec = fondant_component_op.component_spec

            if not load_component:
                # Check subset exists
                for (
                    component_subset_name,
                    component_subset,
                ) in component_spec.consumes.items():
                    if component_subset_name not in manifest.subsets:
                        msg = (
                            f"Component '{component_spec.name}' is trying to invoke the subset "
                            f"'{component_subset_name}', which has not been defined or created "
                            f"in the previous components."
                        )
                        raise InvalidPipelineDefinition(
                            msg,
                        )

                    # Get the corresponding manifest fields
                    manifest_fields = manifest.subsets[component_subset_name].fields

                    # Check fields
                    for field_name, subset_field in component_subset.fields.items():
                        # Check if invoked field exists
                        if field_name not in manifest_fields:
                            msg = (
                                f"The invoked subset '{component_subset_name}' of the "
                                f"'{component_spec.name}' component does not match the "
                                f"previously created subset definition.\n The component is "
                                f"trying to invoke the field '{field_name}' which has not been "
                                f"previously defined. Current available fields are "
                                f"{manifest_fields}\n"
                            )
                            raise InvalidPipelineDefinition(
                                msg,
                            )
                        # Check if the invoked field schema matches the current schema
                        if subset_field != manifest_fields[field_name]:
                            msg = (
                                f"The invoked subset '{component_subset_name}' of the "
                                f"'{component_spec.name}' component does not match  the "
                                f"previously created subset definition.\n The '{field_name}' "
                                f"field is currently defined with the following schema:\n"
                                f"{manifest_fields[field_name]}\nThe current component to "
                                f"trying to invoke it with this schema:\n{subset_field}"
                            )
                            raise InvalidPipelineDefinition(
                                msg,
                            )
            manifest = manifest.evolve(component_spec)
            load_component = False

        logger.info("All pipeline component specifications match.")

    def get_pipeline_cache_dict(
        self,
        cache_disabled: t.Optional[bool] = False,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        """
        Generate a dictionary containing cache information for each component in the pipeline.

        This function iterates over the components in the pipeline and determines whether each
        component should be executed or whether it can be cached based on the cache key.

        Returns:
            dictionary with component names as keys and corresponding cache information as values.
            Each entry in the dictionary contains the 'cache_key' and 'execute_component' fields.

        Example:
            {
                'component1': {
                    'cache_key': 'cache_key_value1',
                    'execute_component': True
                },
                'component2': {
                    'cache_key': 'cache_key_value2',
                    'execute_component': False
                },
                ...
            }
        """
        execute_next_components = False
        cache_dict = {}

        for component_name, component in self._graph.items():
            fondant_component_op = component["fondant_component_op"]

            cache_key = fondant_component_op.get_component_cache_key()

            if cache_disabled:
                execute_component = True
            elif execute_next_components is False:
                execute_component = self.execute_component(
                    component_name=component_name,
                    cache_key=cache_key,
                )
            else:
                execute_component = True

            # if one component should be executed then all subsequent components will have to be
            # executed
            if execute_component is True:
                execute_next_components = True

            # Create cache information entry for the component in the dictionary
            cache_dict[component_name] = {
                "cache_key": cache_key,
                "execute_component": execute_component,
            }

        return cache_dict

    def __repr__(self) -> str:
        """Return a string representation of the FondantPipeline object."""
        return f"{self.__class__.__name__}({self._graph!r}"
