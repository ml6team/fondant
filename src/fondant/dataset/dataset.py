"""This module defines classes to represent a Fondant Dataset."""

import copy
import datetime
import hashlib
import inspect
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

import random

import pyarrow as pa

from fondant.component import BaseComponent
from fondant.core.component_spec import ComponentSpec, OperationSpec
from fondant.core.exceptions import (
    InvalidDatasetDefinition,
    InvalidLightweightComponent,
)
from fondant.core.manifest import Manifest
from fondant.core.schema import Field
from fondant.dataset import Image, LightweightComponent

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
    instance_type: t.Optional[str] = None

    """
    Class representing the resources to assign to a Fondant Component operation in a Fondant
    Dataset.

       Arguments:
           number_of_accelerators: The number of accelerators to assign to the operation (GPU, TPU)
           accelerator_name: The name of the accelerator to assign. If you're using a cluster setup
             on GKE, select "GPU" for GPU or "TPU" for TPU. Make sure
             that you select a nodepool with the available hardware. If you're running the
             dataset materilization workflow on Vertex, then select one of the machines specified
             in the list of accelerators
             here https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec.
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
           instancy_type: the instancy type of the component.
       """

    def __post_init__(self):
        """Validate the resources."""
        if bool(self.node_pool_label) != bool(self.node_pool_name):
            msg = "Both node_pool_label and node_pool_name must be specified or both must be None."
            raise InvalidDatasetDefinition(
                msg,
            )

        if bool(self.accelerator_number) != bool(self.accelerator_name):
            msg = (
                "Both number of accelerators and accelerator name must be specified or both must"
                " be None."
            )
            raise InvalidDatasetDefinition(
                msg,
            )

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Return a dictionary representation of the Resources object."""
        return asdict(self)


class ComponentOp:
    """
    Class representing an operation for a Fondant Component in a Fondant Dataset. An operation
    is a representation of a function that will be executed as part of the workflow.

    Arguments:
        name_or_path: The name of a reusable component, or the path to the directory containing a
            custom component.
        arguments: A dictionary containing the argument name and value for the operation.
        input_partition_rows: The number of rows to load per partition. Set to override the
        automatic partitioning
        resources: The resources to assign to the operation.

    Note:
        - A Fondant Component operation is created by defining a Fondant Component and its input
          arguments.
    """

    COMPONENT_SPEC_NAME = "fondant_component.yaml"

    def __init__(
        self,
        image: Image,
        component_spec: ComponentSpec,
        *,
        consumes: t.Optional[t.Dict[str, str]] = None,
        produces: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[str, int]] = None,
        cache: t.Optional[bool] = True,
        resources: t.Optional[Resources] = None,
        component_dir: t.Optional[Path] = None,
        dataset_fields: t.Optional[t.Mapping[str, Field]] = None,
    ) -> None:
        self.image = image
        self.component_spec = component_spec
        self.input_partition_rows = input_partition_rows
        self.cache = self._configure_caching_from_image_tag(cache)
        self.component_dir = component_dir

        if consumes is None:
            consumes = self._infer_consumes(component_spec, dataset_fields)
        consumes = self._validate_consumes(consumes, component_spec, dataset_fields)

        if produces is None and component_spec.produces_additional_properties:
            logger.warning(
                "Can not infer produces. "
                "The component will not produce any new columns.",
            )

        self.operation_spec = OperationSpec(
            self.component_spec,
            consumes=consumes,
            produces=produces,
        )

        self.arguments = arguments or {}
        self.arguments.update(
            {
                key: value
                for key, value in {
                    "input_partition_rows": input_partition_rows,
                    "cache": self.cache,
                    "operation_spec": self.operation_spec.to_json(),
                }.items()
                if value is not None
            },
        )

        self.arguments.setdefault("operation_spec", self.operation_spec.to_json())

        self.resources = resources or Resources()

    @classmethod
    def from_component_yaml(cls, path, fields=None, **kwargs) -> "ComponentOp":
        if cls._is_custom_component(path):
            component_dir = Path(path)
        else:
            component_dir = cls._get_registry_path(str(path))

        component_spec = ComponentSpec.from_file(
            component_dir / cls.COMPONENT_SPEC_NAME,
        )

        image = Image(
            base_image=component_spec.image,
        )

        return cls(
            image=image,
            component_spec=component_spec,
            component_dir=component_dir,
            dataset_fields=fields,
            **kwargs,
        )

    @classmethod
    def _infer_consumes(
        cls,
        component_spec,
        dataset_fields,
    ) -> t.Union[t.Optional[t.Dict[str, str]], t.Optional[t.Dict[str, pa.DataType]]]:
        """Infer the consumes section of the component spec."""
        if component_spec.consumes_is_defined is False:
            msg = (
                "The consumes section of the component spec is not defined. "
                "Can not infer consumes of the OperationSpec. Please define a consumes section "
                "in the dataset interface. "
            )
            logger.info(msg)
            return None

        # Component has consumes and additionalProperties, we will load all dataset columns
        if (
            component_spec.consumes_is_defined
            and component_spec.consumes_additional_properties
        ):
            if dataset_fields is None:
                logger.info(
                    "The dataset fields are not defined. Cannot infer consumes.",
                )
                return None

            return {k: v.type.value for k, v in dataset_fields.items()}

        # Component has consumes and no additionalProperties, we will load only the columns defined
        # in the component spec
        return {k: v.type.value for k, v in component_spec.consumes.items()}

    @classmethod
    def _validate_consumes(
        cls,
        consumes: t.Optional[t.Dict[str, str]],
        component_spec: ComponentSpec,
        dataset_fields: t.Optional[t.Mapping[str, Field]],
    ) -> t.Union[t.Optional[t.Dict[str, str]], t.Optional[t.Dict[str, pa.DataType]]]:
        """
        Validate the consumes of the component spec.
        Every column in the consumes should be present in the dataset fields and in the
        ComponentSpec. Except if additionalProperties is set to True in the ComponentSpec.
        In that case, we will infer the type from the dataset fields.
        """
        if consumes is None or dataset_fields is None:
            return consumes

        validated_consumes = copy.deepcopy(consumes)

        for operations_column_name, dataset_column_name_or_type in consumes.items():
            # Dataset column name is part of the dataset fields
            if (
                isinstance(dataset_column_name_or_type, str)
                and dataset_column_name_or_type not in dataset_fields.keys()
            ):
                msg = (
                    f"The dataset does not contain the column {dataset_column_name_or_type} "
                    f"required by the component {component_spec.name}."
                )
                raise InvalidDatasetDefinition(msg)

            # If operations column name is not in the component spec, but additional properties
            # are true we will infer the correct type from the dataset fields
            if (
                isinstance(dataset_column_name_or_type, str)
                and operations_column_name not in component_spec.consumes.keys()
            ):
                if component_spec.consumes_additional_properties:
                    validated_consumes[operations_column_name] = dataset_fields[
                        operations_column_name
                    ].type.value
                else:
                    msg = (
                        f"Received a string value for key `{operations_column_name}` in the "
                        f"`consumes` argument passed to the operation, "
                        f"but `{operations_column_name}` is not defined in the `consumes` "
                        f"section of the component spec."
                    )
                    raise InvalidDatasetDefinition(msg)

        return validated_consumes

    @classmethod
    def from_ref(
        cls,
        ref: t.Any,
        fields: t.Optional[t.Mapping[str, Field]] = None,
        **kwargs,
    ) -> "ComponentOp":
        """Create a ComponentOp from a reference. The reference can
        be a reusable component name, a path to a custom component,
        or a python component class.

        Args:
            ref: The name of a reusable component, or the path to the directory containing
                a custom component, or a python component class.
            fields: The fields of the dataset available to the component.
            **kwargs: The provided user arguments are passed in as keyword arguments
        """
        if inspect.isclass(ref) and issubclass(ref, BaseComponent):
            if issubclass(ref, LightweightComponent):
                component_spec = ref.get_component_spec()

                operation = cls(
                    ref.image(),
                    component_spec,
                    dataset_fields=fields,
                    **kwargs,
                )
            else:
                msg = """Reference is not a valid lightweight component.
                       Make sure the component is decorated properly."""
                raise InvalidLightweightComponent(msg)

        elif isinstance(ref, (str, Path)):
            operation = cls.from_component_yaml(
                ref,
                fields,
                **kwargs,
            )
        else:
            msg = f"""Invalid reference type: {type(ref)}.
                Expected a string, Path, or a lightweight component class."""
            raise ValueError(msg)
        return operation

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
                    f"Component `{self.component_spec.name}` has an image tag set to latest. "
                    f"Caching for the component will be disabled to prevent"
                    f" unpredictable behavior due to images updates",
                )
                return False

        return cache

    @property
    def dockerfile_path(self) -> t.Optional[Path]:
        if not self.component_dir:
            return None
        docker_path = self.component_dir / "Dockerfile"
        return docker_path if docker_path.exists() else None

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

    @property
    def component_name(self) -> str:
        return self.component_spec.safe_name

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

        operation_spec_dict = self.operation_spec.to_dict()
        image_dict = self.image.to_dict()

        arguments = (
            get_nested_dict_hash(self.arguments) if self.arguments is not None else None
        )

        component_op_uid_dict = {
            "operation_spec_hash": get_nested_dict_hash(operation_spec_dict),
            "image": get_nested_dict_hash(image_dict),
            "arguments": arguments,
            "input_partition_rows": self.input_partition_rows,
            "number_of_accelerators": self.resources.accelerator_number,
            "accelerator_name": self.resources.accelerator_name,
            "node_pool_name": self.resources.node_pool_name,
        }

        if previous_component_cache is not None:
            component_op_uid_dict["previous_component_cache"] = previous_component_cache

        return get_nested_dict_hash(component_op_uid_dict)


class Dataset:
    def __init__(
        self,
        manifest: Manifest,
        description: t.Optional[str] = None,
    ):
        self.description = description
        self._graph: t.OrderedDict[str, t.Any] = OrderedDict()
        self.task_without_dependencies_added = False
        self.manifest = manifest

    @staticmethod
    def _validate_dataset_name(name: str) -> str:
        pattern = r"^[a-z0-9][a-z0-9_-]*$"
        if not re.match(pattern, name):
            msg = f"The dataset name violates the pattern {pattern}"
            raise InvalidDatasetDefinition(msg)
        return name

    @staticmethod
    def get_run_id(name) -> str:
        """Get a unique run ID for the workspace."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{name}-{timestamp}"

    @property
    def name(self) -> str:
        """The name of the dataset."""
        return self.manifest.dataset_name

    @property
    def package_path(self) -> t.Optional[str]:
        if self.name:
            return f"{self.name}.tgz"
        return None

    def register_operation(
        self,
        operation: ComponentOp,
        *,
        input_dataset: t.Optional["Dataset"],
        output_dataset: t.Optional["Dataset"],
    ) -> None:
        """Register an operation in the dataset graph."""
        dependencies = []
        for component_name, info in self._graph.items():
            if info["output_dataset"] == input_dataset:
                dependencies.append(component_name)

        self._graph[operation.component_name] = {
            "operation": operation,
            "dependencies": dependencies,
            "output_dataset": output_dataset,
        }

    @staticmethod
    def read(manifest_path: str):
        """
        Read a dataset from a manifest file.

        Args:
            manifest_path: The path to the manifest file.
        """
        manifest = Manifest.from_file(manifest_path)
        return Dataset(manifest=manifest)

    @staticmethod
    def create(
        ref: t.Any,
        *,
        produces: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        resources: t.Optional[Resources] = None,
        cache: t.Optional[bool] = True,
        dataset_name: t.Optional[str] = None,
    ) -> "Dataset":
        """
        Read data using the provided component.

        Args:
            ref: The name of a reusable component, or the path to the directory containing
                a containerized component, or a lightweight component class.
            produces: A mapping to update the fields produced by the operation as defined in the
                component spec. The keys are the names of the fields to be received by the
                component, while the values are the type of the field, or the name of the field to
                map from the dataset.
            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            resources: The resources to assign to the operation.
            cache: Set to False to disable caching, True by default.
            dataset_name: The name of the dataset.

        Returns:
            An intermediate dataset.
        """
        if dataset_name is None:
            dataset_name = f"dataset-{random.randint(1, 100)}"  # nosec B311

        operation = ComponentOp.from_ref(
            ref,
            produces=produces,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
        )

        manifest = Manifest.create(
            dataset_name=dataset_name,
            run_id=Dataset.get_run_id(dataset_name),
            component_id=operation.component_name,
        )

        dataset = Dataset(manifest=manifest)

        return dataset._apply(operation)

    def sort_graph(self):
        """Sort the graph topologically based on task dependencies."""
        logger.info("Sorting workflow graph topologically.")
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

    def validate(self):
        """Sort and run validation on the dataset definition.

        Args:
            run_id: run identifier
            workspace: workspace to operate in

        """
        self.sort_graph()
        self._validate_dataset_definition()

    def _validate_dataset_definition(self):
        """
        Validates the dataset definition by ensuring that the consumed and produced subsets and
        their associated fields match and are invoked in the correct order.

        Raises:
            InvalidDatasetDefinition: If a component is trying to invoke a subset that is not
             defined or created in previous components, or if an invoked subset's schema does not
              match the previously created subset definition
        """
        run_id = self.manifest.run_id
        if len(self._graph.keys()) == 0:
            logger.info(
                "No components defined in the dataset workflow. Nothing to validate.",
            )
            return

        load_component = True
        load_component_name = list(self._graph.keys())[0]

        # Create initial manifest
        manifest = Manifest.create(
            run_id=run_id,
            component_id=load_component_name,
            cache_key="42",
        )
        for operation_specs in self._graph.values():
            component_op = operation_specs["operation"]
            operation_spec = component_op.operation_spec

            if not load_component:
                # Check subset exists
                for (
                    component_field_name,
                    component_field,
                ) in operation_spec.consumes_from_dataset.items():
                    if component_field_name not in manifest.fields:
                        msg = (
                            f"Component '{component_op.component_name}' is trying to invoke the"
                            f"field '{component_field_name}', which has not been defined or created"
                            f"in the previous components. \n"
                            f"Available field names: {list(manifest.fields.keys())}"
                        )
                        raise InvalidDatasetDefinition(
                            msg,
                        )

                    # Get the corresponding manifest fields
                    manifest_field = manifest.fields[component_field_name]

                    # Check if the invoked field schema matches the current schema
                    if component_field.type != manifest_field.type:
                        msg = (
                            f"The invoked field '{component_field_name}' of the "
                            f"'{component_op.component_name}' component does not match the "
                            f"previously created field type.\n The '{manifest_field.name}' "
                            f"field is currently defined with the following type:\n"
                            f"{manifest_field.type}\nThe current component to "
                            f"trying to invoke it with this type:\n{component_field.type}"
                        )
                        raise InvalidDatasetDefinition(
                            msg,
                        )

            # Note: the manifest created here does not have to contain a valid working dir. The
            # manifest information are only used for validation during.
            manifest = manifest.evolve(
                operation_spec,
                run_id=run_id,
                working_directory="dummy-dir",
            )
            load_component = False

        logger.info("All workflow component specifications match.")

    def __repr__(self) -> str:
        """Return a string representation of the Fondant dataset object."""
        return f"{self.__class__.__name__}({self._graph!r}"

    @property
    def fields(self) -> t.Mapping[str, Field]:
        """The fields of the manifest as an immutable mapping."""
        return dict(self.manifest.fields)

    def _apply(self, operation: ComponentOp) -> "Dataset":
        """
        Apply the provided operation to the dataset.

        Args:
            operation: The operation to apply.
            workspace: The workspace to operate in.
        """
        if self.manifest is None:
            msg = "No manifest found."
            raise ValueError(msg)

        evolved_manifest = self.manifest.evolve(
            operation.operation_spec,
            run_id=Dataset.get_run_id(self.name),
        )

        evolved_dataset = Dataset(
            manifest=evolved_manifest,
        )

        evolved_dataset._graph = self._graph

        evolved_dataset.register_operation(
            operation,
            input_dataset=self,  # using reference to manifests instead?
            output_dataset=evolved_dataset,
        )

        return evolved_dataset

    def apply(
        self,
        ref: t.Any,
        *,
        consumes: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        produces: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        resources: t.Optional[Resources] = None,
        cache: t.Optional[bool] = True,
    ) -> "Dataset":
        """
        Apply the provided component on the dataset.

        Args:
            ref: The name of a reusable component, or the path to the directory containing
                a custom component, or a lightweight component class.
            workspace: workspace to operate in
            consumes: A mapping to update the fields consumed by the operation as defined in the
                component spec. The keys are the names of the fields to be received by the
                component, while the values are the type of the field, or the name of the field to
                map from the input dataset.

                Suppose we have a component spec that expects the following fields:

                ```
                ...
                consumes:
                    text:
                        type: string
                    image:
                        type: binary
                ...
                ```

                To override the default mapping and specify that the 'text' field should be sourced
                from the 'custom_text' field in the input dataset, the 'consumes' mapping can be
                defined as follows:

                ```
                consumes = {
                    "text": "custom_text"
                }
                ```

                In this example, the 'text' field will be sourced from 'custom_text' and 'image'
                will be sourced from the 'image' field by default, since it's not specified in the
                custom mapping.

            produces: A mapping to update the fields produced by the operation as defined in the
                component spec. The keys are the names of the fields to be produced by the
                component, while the values are the type of the field, or the name that should be
                used to write the field to the output dataset.

                Suppose we have a component spec that expects the following fields:

                ```
                ...
                produces:
                    text:
                        type: string
                    width:
                        type: int
                ```

                To customize the field names and types during the production step, the 'produces'
                mapping can be defined as follows:

                ```
                produces = {
                    "width": "custom_width",
                }
                ```

                In this example, the 'text' field will retain as text since it is not specified in
                the custom mapping. The 'width' field will be stored with the name 'custom_width'
                in the output dataset.

                Alternatively, the produces defines the data type of the output data.

                ```
                produces = {
                    "width": pa.float32(),
                }
                ```

                In this example, the 'text' field will retain its type 'string' without specifying a
                different source, while the 'width' field will be produced as type `float` in the
                output dataset.

            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            resources: The resources to assign to the operation.
            cache: Set to False to disable caching, True by default.

        Returns:
            An intermediate dataset.
        """
        operation = ComponentOp.from_ref(
            ref,
            fields=self.fields,
            produces=produces,
            consumes=consumes,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
        )

        return self._apply(operation)

    def write(
        self,
        ref: t.Any,
        *,
        consumes: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        resources: t.Optional[Resources] = None,
        cache: t.Optional[bool] = True,
    ) -> "Dataset":
        """
        Write the dataset using the provided component.

        Args:
            ref: The name of a reusable component, or the path to the directory containing
                a custom component, or a lightweight component class.
            workspace: workspace to operate in
            consumes: A mapping to update the fields consumed by the operation as defined in the
                component spec. The keys are the names of the fields to be received by the
                component, while the values are the type of the field, or the name of the field to
                map from the input dataset.
            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            resources: The resources to assign to the operation.
            cache: Set to False to disable caching, True by default.

        Returns:
            An intermediate dataset.
        """
        # TODO: add method call to retrieve workspace context, and make passing workspace optional

        operation = ComponentOp.from_ref(
            ref,
            fields=self.fields,
            consumes=consumes,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
        )
        return self._apply(operation)
