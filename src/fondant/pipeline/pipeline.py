"""This module defines classes to represent a Fondant Pipeline."""
import copy
import datetime
import hashlib
import json
import logging
import re
import typing as t
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

import pyarrow as pa

from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidPipelineDefinition
from fondant.core.manifest import Manifest
from fondant.core.schema import Field, ProducesType, Type

logger = logging.getLogger(__name__)

VALID_ACCELERATOR_TYPES = [
    "GPU",
    "TPU",
]

# Taken from https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform_v1/types
# /accelerator_type.py
VALID_VERTEX_ACCELERATOR_TYPES = [
    "ACCELERATOR_TYPE_UNSPECIFIED",
    "NVIDIA_TESLA_K80",
    "NVIDIA_TESLA_P100",
    "NVIDIA_TESLA_V100",
    "NVIDIA_TESLA_P4",
    "NVIDIA_TESLA_T4",
    "NVIDIA_TESLA_A100",
    "NVIDIA_A100_80GB",
    "NVIDIA_L4",
    "TPU_V2",
    "TPU_V3",
    "TPU_V4_POD",
]


@dataclass
class Resources:
    accelerator_name: t.Optional[str] = None
    accelerator_number: t.Optional[int] = None
    cpu_request: t.Optional[str] = None
    cpu_limit: t.Optional[str] = None
    memory_request: t.Optional[str] = None
    memory_limit: t.Optional[str] = None
    node_pool_label: t.Optional[str] = None
    node_pool_name: t.Optional[str] = None

    """
    Class representing the resources to assign to a Fondant Component operation in a Fondant
    Pipeline.

       Arguments:
           number_of_accelerators: The number of accelerators to assign to the operation (GPU, TPU)
           accelerator_name: The name of the accelerator to assign. If you're using a cluster setup
             on GKE, select "GPU" for GPU or "TPU" for TPU. Make sure
             that you select a nodepool with the available hardware. If you're running the
             pipeline on Vertex, then select one of the machines specified in the list of
             accelerators here https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec.
           node_pool_label: The label of the node pool to which the operation will be assigned.
           node_pool_name: The name of the node pool to which the operation will be assigned.
           cache: Set to False to disable caching, True by default.
           cpu_request: the memory requested by the component. The value
            should be a string which can be a number or a number followed by “m”, which means
            1/1000.
           cpu_limit: the maximum amount of CPU that can be used by the component. The value
            should be a string which can be a number or a number followed by “m”, which means
            1/1000.
           memory_request: the memory requested by the component. The value  can be a number or a
             number followed by one of “E”, “P”, “T”, “G”, “M”, “K”.
           memory_limit: the maximum memory that can be used by the component. The value  can be a
            number or a number followed by one of “E”, “P”, “T”, “G”, “M”, “K”.
       """

    def __post_init__(self):
        """Validate the resources."""
        if bool(self.node_pool_label) != bool(self.node_pool_name):
            msg = "Both node_pool_label and node_pool_name must be specified or both must be None."
            raise InvalidPipelineDefinition(
                msg,
            )

        if bool(self.accelerator_number) != bool(self.accelerator_name):
            msg = (
                "Both number of accelerators and accelerator name must be specified or both must"
                " be None."
            )
            raise InvalidPipelineDefinition(
                msg,
            )

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Return a dictionary representation of the Resources object."""
        return asdict(self)


class ComponentOp:
    """
    Class representing an operation for a Fondant Component in a Fondant Pipeline. An operation
    is a representation of a function that will be executed as part of a pipeline.

    Arguments:
        name_or_path: The name of a reusable component, or the path to the directory containing a
            custom component.
        arguments: A dictionary containing the argument name and value for the operation.
        input_partition_rows: The number of rows to load per partition. Set to override the
        automatic partitioning
        resources: The resources to assign to the operation.
        cluster_type: The type of cluster to use for distributed execution (default is "local").
        client_kwargs: Keyword arguments used to initialise the dask client.

    Note:
        - A Fondant Component operation is created by defining a Fondant Component and its input
          arguments.
    """

    COMPONENT_SPEC_NAME = "fondant_component.yaml"

    def __init__(
        self,
        name_or_path: t.Union[str, Path],
        *,
        consumes: t.Optional[t.Dict[str, str]] = None,
        produces: t.Optional[
            t.Union[t.Dict[str, pa.DataType], t.Dict[str, str]]
        ] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[str, int]] = None,
        cache: t.Optional[bool] = True,
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
        resources: t.Optional[Resources] = None,
    ) -> None:
        if self._is_custom_component(name_or_path):
            self.component_dir = Path(name_or_path)
        else:
            self.component_dir = self._get_registry_path(str(name_or_path))

        self.input_partition_rows = input_partition_rows
        self.component_spec = ComponentSpec.from_file(
            self.component_dir / self.COMPONENT_SPEC_NAME,
        )
        self.name = self.component_spec.component_folder_name
        self.cache = self._configure_caching_from_image_tag(cache)
        self.cluster_type = cluster_type
        self.client_kwargs = client_kwargs
        self.consumes = consumes
        self.produces = self._produces_parser(produces)

        self.arguments = arguments or {}
        self._add_component_argument("input_partition_rows", self.input_partition_rows)
        self._add_component_argument("cache", self.cache)
        self._add_component_argument("cluster_type", self.cluster_type)
        self._add_component_argument("client_kwargs", self.client_kwargs)
        self._add_component_argument("consumes", self.consumes)
        self._add_component_argument("produces", self.produces)
        self.arguments.setdefault("component_spec", self.component_spec.specification)

        self.resources = resources or Resources()

    @staticmethod
    def _produces_parser(
        produces: t.Optional[ProducesType] = None,
    ) -> t.Union[None, t.Dict[str, t.Any]]:
        """
        Parse the produces argument to a valid string representation.

        Args:
            produces: The produces argument to parse.

        Returns:
            The parsed produces argument as a string representation.
        """
        if produces is None:
            return None

        parsed_produces: t.Dict[str, t.Any] = {}

        for column_name, mapping_name_or_type in produces.items():
            if isinstance(mapping_name_or_type, pa.DataType):
                parsed_produces[column_name] = Type(mapping_name_or_type).to_json()
            elif isinstance(mapping_name_or_type, str):
                parsed_produces[column_name] = mapping_name_or_type
            else:
                msg = (
                    "The produces argument must be a dictionary with column names as keys and"
                    " mapping names or pyarrow data types as values."
                )
                raise InvalidPipelineDefinition(
                    msg,
                )
        return parsed_produces

    def _configure_caching_from_image_tag(
        self,
        cache: t.Optional[bool],
    ) -> t.Optional[bool]:
        """
        Adjusts the caching setting based on the image tag of the component.

        If the `cache` parameter is set to `True`, this function checks the image tag of
        the component and disables caching (`cache` set to `False`) if the image tag is "latest".
        This is because using "latest" image tags can lead to unpredictable behavior due to
         image updates.

        Args:
            cache: The current caching setting. Set to `True` to enable caching, `False` to disable.

        Returns:
            The adjusted caching setting based on the image tag.

        """
        if cache is True:
            image_tag = self.component_spec.image.rsplit(":")[-1]

            if image_tag == "latest":
                logger.warning(
                    f"Component `{self.name}` has an image tag set to latest. "
                    f"Caching for the component will be disabled to prevent"
                    f" unpredictable behavior due to images updates",
                )
                return False

        return cache

    def _add_component_argument(
        self,
        argument_name: str,
        argument_value: t.Any,
    ):
        """Register component argument to arguments dict as well as component attributes."""
        if hasattr(self, "arguments") is False:
            self.arguments = {}

        if argument_value is not None:
            self.arguments[argument_name] = argument_value

    @property
    def dockerfile_path(self) -> t.Optional[Path]:
        path = self.component_dir / "Dockerfile"
        return path if path.exists() else None

    @staticmethod
    def _is_custom_component(path_or_name: t.Union[str, Path]) -> bool:
        """Checks if name is a local path and a custom component."""
        component_dir: Path = Path(path_or_name)
        return component_dir.exists() and component_dir.is_dir()

    @staticmethod
    def _get_registry_path(name: str) -> Path:
        """Checks if name is a local path and a custom component."""
        component_dir: Path = t.cast(Path, files("fondant") / f"components/{name}")
        if not (component_dir.exists() and component_dir.is_dir()):
            msg = f"No reusable component with name {name} found."
            raise ValueError(msg)
        return component_dir

    def get_component_cache_key(
        self,
        previous_component_cache: t.Optional[str] = None,
    ) -> str:
        """Calculate a cache key representing the unique identity of this ComponentOp.

        The cache key is computed based on the component specification, image hash, arguments, and
        other attributes of the ComponentOp. It is used to uniquely identify a specific instance
        of the ComponentOp and is used for caching.

        Returns:
            A cache key representing the unique identity of this ComponentOp.
        """

        def get_nested_dict_hash(input_dict):
            """Calculate the hash of a nested dictionary.

            Args:
                input_dict: The nested dictionary to calculate the hash for.

            Returns:
                The hash value (MD5 digest) of the nested dictionary.
            """
            sorted_json_string = json.dumps(input_dict, sort_keys=True)
            hash_object = hashlib.md5(sorted_json_string.encode())  # nosec
            return hash_object.hexdigest()

        component_spec_dict = self.component_spec.specification
        arguments = (
            get_nested_dict_hash(self.arguments) if self.arguments is not None else None
        )

        component_op_uid_dict = {
            "component_spec_hash": get_nested_dict_hash(component_spec_dict),
            "arguments": arguments,
            "input_partition_rows": self.input_partition_rows,
            "number_of_accelerators": self.resources.accelerator_number,
            "accelerator_name": self.resources.accelerator_name,
            "node_pool_name": self.resources.node_pool_name,
        }

        if previous_component_cache is not None:
            component_op_uid_dict["previous_component_cache"] = previous_component_cache

        return get_nested_dict_hash(component_op_uid_dict)


class Pipeline:
    """Class representing a Fondant Pipeline."""

    def __init__(
        self,
        name: str,
        *,
        base_path: str,
        description: t.Optional[str] = None,
    ):
        """
        Args:
            name: The name of the pipeline.
            base_path: The base path for the pipeline where the artifacts are stored.
            description: Optional description of the pipeline.
        """
        self.base_path = base_path
        self.name = self._validate_pipeline_name(name)
        self.description = description
        self.package_path = f"{name}.tgz"
        self._graph: t.OrderedDict[str, t.Any] = OrderedDict()
        self.task_without_dependencies_added = False

    def _apply(
        self,
        operation: ComponentOp,
        datasets: t.Optional[t.Union["Dataset", t.List["Dataset"]]] = None,
    ) -> "Dataset":
        """
        Apply an operation to the provided input datasets.

        Args:
            operation: The operation to apply.
            datasets: The input datasets to apply the operation on
        """
        if datasets is None:
            datasets = []
        elif not isinstance(datasets, list):
            datasets = [datasets]

        if len(datasets) > 1:
            msg = (
                f"Multiple input datasets provided for operation `{operation.name}`. The current "
                f"version of Fondant can only handle components with a single input."
            )
            raise InvalidPipelineDefinition(
                msg,
            )

        self._graph[operation.name] = {
            "operation": copy.deepcopy(operation),
            "dependencies": [dataset.operation.name for dataset in datasets],
        }
        return Dataset(pipeline=self, operation=operation)

    def read(
        self,
        name_or_path: t.Union[str, Path],
        *,
        produces: t.Dict[str, pa.DataType],
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        resources: t.Optional[Resources] = None,
        cache: t.Optional[bool] = True,
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
    ) -> "Dataset":
        """Read data using the provided operation.


        Arguments:
            name_or_path: The name of a hub component or a path to a custom component directory.
            produces: a dictionary of the fields to be produced by the component. The keys are the
             names of the dataset fields. The values are the data type of the field to define the
             schema of generic components
            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            resources: The resources to assign to the operation.
            cluster_type: The type of cluster to use for distributed execution (default is "local").
            client_kwargs: Keyword arguments used to initialise the dask client.
            cache: Set to False in order to disable caching, True by default.
        """
        if self._graph:
            msg = "For now, at most one read component can be applied per pipeline."
            raise InvalidPipelineDefinition(
                msg,
            )

        operation = ComponentOp(
            name_or_path,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
            cluster_type=cluster_type,
            client_kwargs=client_kwargs,
            produces=produces,
        )
        return self._apply(operation)

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

    def get_run_id(self) -> str:
        """Get a unique run ID for the pipeline."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.name}-{timestamp}"

    def validate(self, run_id: str):
        """Sort and run validation on the pipeline definition.

        Args:
            run_id: run identifier

        """
        self.sort_graph()
        self._validate_pipeline_definition(run_id)

    def _validate_pipeline_definition(  # ruff: noqa: PLR0912
        self,
        run_id: str,
    ):
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

        def _validate_field_schema(manifest_field: Field, component_field: Field):
            """Validate that the field schema matches."""
            if component_field.type != manifest_field.type:
                error_msg = (
                    f"The invoked field '{component_field.name}' of the "
                    f"'{component_spec.name}' component does not match the "
                    f"previously created field type.\n The '{manifest_field.name}' "
                    f"field is currently defined with the following type:\n"
                    f"{manifest_field.type}\nThe current component to "
                    f"trying to invoke it with this type:\n{component_field.type}"
                )
                raise InvalidPipelineDefinition(
                    error_msg,
                )

        if len(self._graph.keys()) == 0:
            logger.info("No components defined in the pipeline. Nothing to validate.")
            return

        # TODO: change later if we decide to run 2 fondant pipelines after each other
        load_component = True
        load_component_name = list(self._graph.keys())[0]

        # Create initial manifest
        manifest = Manifest.create(
            pipeline_name=self.name,
            base_path=self.base_path,
            run_id=run_id,
            component_id=load_component_name,
            cache_key="42",
        )
        for operation_specs in self._graph.values():
            fondant_component_op = operation_specs["operation"]
            component_spec = fondant_component_op.component_spec
            component_op_consumes = fondant_component_op.consumes

            if component_op_consumes:
                # Get reverse mapping of component_op_consumes (component column-> dataset column)
                component_op_consumes = {
                    value: key for key, value in component_op_consumes.items()
                }

            if not load_component:
                for (
                    component_field_name,
                    component_field,
                ) in component_spec.consumes.items():
                    dataset_field_name = component_field_name
                    # Check if the invoked field is defined in the manifest
                    if dataset_field_name not in manifest.fields:
                        # Check if there exist a custom consumes mapping for the field
                        if component_op_consumes:
                            if (
                                component_field_name in component_op_consumes
                                and component_op_consumes[component_field_name]
                                in manifest.fields
                            ):
                                dataset_field_name = component_op_consumes.pop(
                                    component_field_name,
                                )
                            else:
                                msg = (
                                    f"Component '{component_spec.name}' is trying to invoke the"
                                    f" field "
                                    f"'{component_field_name}', which has not been defined "
                                    f"or created "
                                    f"in the previous components."
                                )
                                raise InvalidPipelineDefinition(msg)
                        else:
                            msg = (
                                f"Component '{component_spec.name}' is trying to invoke the field "
                                f"'{component_field_name}', which has not been defined or created "
                                f"in the previous components."
                            )
                            raise InvalidPipelineDefinition(msg)
                    _validate_field_schema(
                        manifest.fields[dataset_field_name],
                        component_spec.consumes[component_field_name],
                    )

                # Check for remaining custom consumes mappings
                if component_op_consumes:
                    if not component_spec.is_consumes_generic:
                        msg = (
                            f"Found fields {component_op_consumes} in custom consumes "
                            f"mapping for"
                            f" component '{component_spec.name}'"
                            f" which is not a generic component that does not have a "
                            f"corresponding field in the component spec."
                        )
                        raise InvalidPipelineDefinition(msg)

                    for _, dataset_column in component_op_consumes.items():
                        if dataset_column not in manifest.fields:
                            msg = (
                                f"Generic Component '{component_spec.name}' is trying to invoke"
                                f" the field '{dataset_column}', which has not been defined or "
                                f"created "
                                f"in the previous components."
                            )
                            raise InvalidPipelineDefinition(msg)

            manifest = manifest.evolve(
                component_spec,
                run_id=run_id,
                produces=fondant_component_op.produces,
            )
            load_component = False

        logger.info("All pipeline component specifications match.")

    def __repr__(self) -> str:
        """Return a string representation of the FondantPipeline object."""
        return f"{self.__class__.__name__}({self._graph!r}"


class Dataset:
    def __init__(self, *, pipeline: Pipeline, operation: ComponentOp) -> None:
        """A class representing an intermediate dataset.

        Args:
            pipeline: The pipeline this dataset is a part of.
            operation: The operation that created this dataset.
        """
        self.pipeline = pipeline
        self.operation = operation

    def apply(
        self,
        name_or_path: t.Union[str, Path],
        *,
        consumes: t.Optional[t.Dict[str, str]] = None,
        produces: t.Optional[ProducesType] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        resources: t.Optional[Resources] = None,
        cache: t.Optional[bool] = True,
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
    ) -> "Dataset":
        """
        Dataset operation to write the final output of the pipeline.

        Arguments:
            name_or_path: The name of a hub component or a path to a custom component directory.
            consumes: a dictionary of the fields to be written by the component. The keys are the
             names of the dataset fields and the values are the name of the columns to map to.
            produces: a dictionary of the fields to be produced by the component. The keys are the
                names of the dataset fields. The values can be either:
                - A string representing the name of the column to map to for non-generic components
                - The data type of the field produces that will be produced  for generic
                components
            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            resources: The resources to assign to the operation.
            cluster_type: The type of cluster to use for distributed execution (default is "local").
            client_kwargs: Keyword arguments used to initialise the dask client.
            cache: Set to False in order to disable caching, True by default.
        """
        operation = ComponentOp(
            name_or_path,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
            cluster_type=cluster_type,
            client_kwargs=client_kwargs,
            consumes=consumes,
            produces=produces,
        )
        return self.pipeline._apply(operation, self)

    def write(
        self,
        name_or_path: t.Union[str, Path],
        *,
        consumes: t.Optional[t.Dict[str, str]] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        resources: t.Optional[Resources] = None,
        cache: t.Optional[bool] = True,
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
    ) -> None:
        """
        Dataset operation to write the final output of the pipeline.

        Arguments:
            name_or_path: The name of a hub component or a path to a custom component directory.
            consumes: a dictionary of the fields to be written by the component. The keys are the
             names of the dataset fields and the values are the name of the columns to map to.
            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            resources: The resources to assign to the operation.
            cluster_type: The type of cluster to use for distributed execution (default is "local").
            client_kwargs: Keyword arguments used to initialise the dask client.
            cache: Set to False in order to disable caching, True by default.
        """
        operation = ComponentOp(
            name_or_path,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
            cluster_type=cluster_type,
            client_kwargs=client_kwargs,
            consumes=consumes,
        )
        self.pipeline._apply(operation, self)
