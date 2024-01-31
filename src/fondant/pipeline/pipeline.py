"""This module defines classes to represent a Fondant Pipeline."""
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

import pyarrow as pa

from fondant.component import BaseComponent
from fondant.core.component_spec import ComponentSpec, OperationSpec
from fondant.core.exceptions import (
    InvalidLightweightComponent,
    InvalidPipelineDefinition,
)
from fondant.core.manifest import Manifest
from fondant.core.schema import Field
from fondant.pipeline import Image, LightweightComponent
from fondant.pipeline.argument_inference import infer_arguments

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
           instancy_type: the instancy type of the component.
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
        image: Image,
        component_spec: ComponentSpec,
        *,
        consumes: t.Optional[t.Dict[str, str]] = None,
        produces: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[str, int]] = None,
        cache: t.Optional[bool] = True,
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
        resources: t.Optional[Resources] = None,
        component_dir: t.Optional[Path] = None,
    ) -> None:
        self.image = image
        self.component_spec = component_spec
        self.input_partition_rows = input_partition_rows
        self.cache = self._configure_caching_from_image_tag(cache)
        self.cluster_type = cluster_type
        self.client_kwargs = client_kwargs
        self.component_dir = component_dir

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
                    "cluster_type": cluster_type,
                    "client_kwargs": client_kwargs,
                    "operation_spec": self.operation_spec.to_json(),
                }.items()
                if value is not None
            },
        )

        self.arguments.setdefault("operation_spec", self.operation_spec.to_json())

        self.resources = resources or Resources()

    @classmethod
    def from_component_yaml(cls, path, **kwargs) -> "ComponentOp":
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
            **kwargs,
        )

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
                name = ref.__name__
                image = ref.image()
                description = ref.__doc__ or "lightweight component"
                spec_produces = ref.get_spec_produces()

                spec_consumes = (
                    ref.get_spec_consumes(fields, kwargs["consumes"])
                    if fields
                    else {"additionalProperties": True}
                )

                component_spec = ComponentSpec(
                    name,
                    image.base_image,
                    description=description,
                    consumes=spec_consumes,
                    produces=spec_produces,
                    args={
                        name: arg.to_spec()
                        for name, arg in infer_arguments(ref).items()
                    },
                )

                operation = cls(
                    image,
                    component_spec,
                    **kwargs,
                )
            else:
                msg = """Reference is not a valid lightweight component.
                       Make sure the component is decorated properly."""
                raise InvalidLightweightComponent(msg)

        elif isinstance(ref, (str, Path)):
            operation = cls.from_component_yaml(
                ref,
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
            "cluster_type": self.cluster_type,
            "client_kwargs": self.client_kwargs,
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
            base_path: The base path for the pipeline to use to store artifacts and data. This
                can be a local path or a remote path on one of the supported cloud storage
                services. The path should already exist.
            description: Optional description of the pipeline.
        """
        self.base_path = base_path
        self.name = self._validate_pipeline_name(name)
        self.description = description
        self.package_path = f"{name}.tgz"
        self._graph: t.OrderedDict[str, t.Any] = OrderedDict()
        self.task_without_dependencies_added = False

    def register_operation(
        self,
        operation: ComponentOp,
        *,
        input_dataset: t.Optional["Dataset"],
        output_dataset: t.Optional["Dataset"],
    ) -> None:
        dependencies = []
        for component_name, info in self._graph.items():
            if info["output_dataset"] == input_dataset:
                dependencies.append(component_name)

        self._graph[operation.component_name] = {
            "operation": operation,
            "dependencies": dependencies,
            "output_dataset": output_dataset,
        }

    def read(
        self,
        ref: t.Any,
        *,
        produces: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        input_partition_rows: t.Optional[t.Union[int, str]] = None,
        resources: t.Optional[Resources] = None,
        cache: t.Optional[bool] = True,
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
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
            cluster_type: The type of cluster to use for distributed execution (default is "local").
            client_kwargs: Keyword arguments used to initialise the Dask client.

        Returns:
            An intermediate dataset.
        """
        if self._graph:
            msg = "For now, at most one read component can be applied per pipeline."
            raise InvalidPipelineDefinition(
                msg,
            )

        operation = ComponentOp.from_ref(
            ref,
            produces=produces,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
            cluster_type=cluster_type,
            client_kwargs=client_kwargs,
        )
        manifest = Manifest.create(
            pipeline_name=self.name,
            base_path=self.base_path,
            run_id=self.get_run_id(),
            component_id=operation.component_name,
        )
        dataset = Dataset(manifest, pipeline=self)

        return dataset._apply(operation)

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
            pipeline_name=self.name,
            base_path=self.base_path,
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
                ) in operation_spec.outer_consumes.items():
                    if component_field_name not in manifest.fields:
                        msg = (
                            f"Component '{component_op.component_name}' is trying to invoke the"
                            f"field '{component_field_name}', which has not been defined or created"
                            f"in the previous components. \n"
                            f"Available field names: {list(manifest.fields.keys())}"
                        )
                        raise InvalidPipelineDefinition(
                            msg,
                        )

                    # Get the corresponding manifest fields
                    manifest_field = manifest.fields[component_field_name]

                    # Check if the invoked field schema matches the current schema
                    if component_field.type != manifest_field.type:
                        msg = (
                            f"The invoked field '{component_field_name}' of the "
                            f"'{component_op.name}' component does not match  the "
                            f"previously created field type.\n The '{manifest_field.name}' "
                            f"field is currently defined with the following type:\n"
                            f"{manifest_field.type}\nThe current component to "
                            f"trying to invoke it with this type:\n{component_field.type}"
                        )
                        raise InvalidPipelineDefinition(
                            msg,
                        )

            manifest = manifest.evolve(operation_spec, run_id=run_id)
            load_component = False

        logger.info("All pipeline component specifications match.")

    def __repr__(self) -> str:
        """Return a string representation of the FondantPipeline object."""
        return f"{self.__class__.__name__}({self._graph!r}"


class Dataset:
    def __init__(self, manifest, *, pipeline: Pipeline) -> None:
        """A class representing an intermediate dataset.

        Args:
            manifest: Manifest representing the dataset
            pipeline: The pipeline this dataset is a part of.
        """
        self.manifest = manifest
        self.pipeline = pipeline

    @property
    def fields(self) -> t.Mapping[str, Field]:
        """The fields of the manifest as an immutable mapping."""
        return dict(self.manifest.fields)

    def _apply(self, operation: ComponentOp) -> "Dataset":
        """
        Apply the provided operation to the dataset.

        Args:
            operation: The operation to apply.
        """
        evolved_manifest = self.manifest.evolve(
            operation.operation_spec,
            run_id=self.pipeline.get_run_id(),
        )
        evolved_dataset = Dataset(evolved_manifest, pipeline=self.pipeline)

        if self.pipeline is not None:
            self.pipeline.register_operation(
                operation,
                input_dataset=self,
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
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
    ) -> "Dataset":
        """
        Apply the provided component on the dataset.

        Args:
            ref: The name of a reusable component, or the path to the directory containing
                a custom component, or a lightweight component class.
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
            cluster_type: The type of cluster to use for distributed execution (default is "local").
            client_kwargs: Keyword arguments used to initialise the Dask client.

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
            cluster_type=cluster_type,
            client_kwargs=client_kwargs,
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
        cluster_type: t.Optional[str] = "default",
        client_kwargs: t.Optional[dict] = None,
    ) -> None:
        """
        Write the dataset using the provided component.

        Args:
            ref: The name of a reusable component, or the path to the directory containing
                a custom component, or a lightweight component class.
            consumes: A mapping to update the fields consumed by the operation as defined in the
                component spec. The keys are the names of the fields to be received by the
                component, while the values are the type of the field, or the name of the field to
                map from the input dataset.
            arguments: A dictionary containing the argument name and value for the operation.
            input_partition_rows: The number of rows to load per partition. Set to override the
            automatic partitioning
            resources: The resources to assign to the operation.
            cache: Set to False to disable caching, True by default.
            cluster_type: The type of cluster to use for distributed execution (default is "local").
            client_kwargs: Keyword arguments used to initialise the Dask client.

        Returns:
            An intermediate dataset.
        """
        operation = ComponentOp.from_ref(
            ref,
            fields=self.fields,
            consumes=consumes,
            arguments=arguments,
            input_partition_rows=input_partition_rows,
            resources=resources,
            cache=cache,
            cluster_type=cluster_type,
            client_kwargs=client_kwargs,
        )
        self._apply(operation)
