"""This module defines classes to represent an Fondant component specification."""
import copy
import json
import pkgutil
import re
import types
import typing as t
from dataclasses import dataclass
from pathlib import Path

import jsonschema.exceptions
import pyarrow as pa
import yaml
from jsonschema import Draft4Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

from fondant.core.exceptions import InvalidComponentSpec, InvalidPipelineDefinition
from fondant.core.schema import Field, Type


@dataclass
class Argument:
    """
    Component argument.

    Args:
        name: name of the argument
        description: argument description
        type: the python argument type in str format (str, int, ...)
        default: default value of the argument (defaults to None)
        optional: whether an argument is optional or not (defaults to False)
    """

    name: str
    description: str
    type: str
    default: t.Any = None
    optional: t.Optional[bool] = False

    def __post_init__(self):
        self.default = None if self.default == "None" else self.default

    @property
    def python_type(self) -> t.Any:
        lookup = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "dict": json.loads,
            "list": json.loads,
        }
        map_fn = lookup[self.type]
        return lambda value: map_fn(value) if value != "None" else None  # type: ignore

    @property
    def kubeflow_type(self) -> str:
        lookup = {
            "str": "STRING",
            "int": "NUMBER_INTEGER",
            "float": "NUMBER_DOUBLE",
            "bool": "BOOLEAN",
            "dict": "STRUCT",
            "list": "LIST",
        }
        return lookup[self.type]


class ComponentSpec:
    """
    Class representing a Fondant component specification.

    Args:
        specification: The fondant component specification as a Python dict
    """

    def __init__(self, specification: t.Dict[str, t.Any]) -> None:
        self._specification = copy.deepcopy(specification)
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate a component specification against the component schema.

        Raises: InvalidComponent when the component specification is not valid.
        """
        spec_data = pkgutil.get_data("fondant", "core/schemas/component_spec.json")

        if spec_data is None:
            msg = "component_spec.json not found in fondant schema"
            raise FileNotFoundError(msg)

        spec_str = spec_data.decode("utf-8")
        spec_schema = json.loads(spec_str)

        base_uri = Path(__file__).parent / "schemas"

        def retrieve_from_filesystem(uri: str) -> Resource:
            path = base_uri / uri
            contents = json.loads(path.read_text())
            return Resource.from_contents(contents, default_specification=DRAFT4)

        registry = Registry(retrieve=retrieve_from_filesystem)  # type: ignore
        validator = Draft4Validator(spec_schema, registry=registry)  # type: ignore

        try:
            validator.validate(self._specification)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidComponentSpec.create_from(e)

    @classmethod
    def from_file(cls, path: t.Union[str, Path]) -> "ComponentSpec":
        """Load the component spec from the file specified by the provided path."""
        with open(path, encoding="utf-8") as file_:
            specification = yaml.safe_load(file_)
            return cls(specification)

    def to_file(self, path) -> None:
        """Dump the component spec to the file specified by the provided path."""
        with open(path, "w", encoding="utf-8") as file_:
            yaml.dump(self._specification, file_)

    @property
    def name(self):
        return self._specification["name"]

    @property
    def component_folder_name(self):
        """Cleans and converts a name to a proper folder name."""
        return self._specification["name"].lower().replace(" ", "_")

    @property
    def sanitized_component_name(self):
        """Cleans and converts a name to be kfp V2 compatible.

        Taken from https://github.com/kubeflow/pipelines/blob/
        cfe671c485d4ee8514290ee81ca2785e8bda5c9b/sdk/python/kfp/dsl/utils.py#L52
        """
        return (
            re.sub(
                "-+",
                "-",
                re.sub("[^-0-9a-z]+", "-", self._specification["name"].lower()),
            )
            .lstrip("-")
            .rstrip("-")
        )

    @property
    def description(self):
        return self._specification["description"]

    @property
    def image(self) -> str:
        return self._specification["image"]

    @image.setter
    def image(self, value: str) -> None:
        self._specification["image"] = value

    @property
    def tags(self) -> t.List[str]:
        return self._specification.get("tags", None)

    @property
    def consumes(self) -> t.Mapping[str, Field]:
        """The fields consumed by the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(name=name, type=Type.from_json(field))
                for name, field in self._specification.get("consumes", {}).items()
                if name != "additionalProperties"
            },
        )

    @property
    def produces(self) -> t.Mapping[str, Field]:
        """The fields produced by the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(name=name, type=Type.from_json(field))
                for name, field in self._specification.get("produces", {}).items()
                if name != "additionalProperties"
            },
        )

    def is_generic(self, mapping: str) -> bool:
        """Returns a boolean indicating whether the provided mapping is generic.

        Args:
            mapping: "consumes" or "produces"
        """
        additional_fields = self._specification.get(mapping, {}).get(
            "additionalProperties",
        )

        return bool(additional_fields)

    @property
    def previous_index(self) -> t.Optional[str]:
        return self._specification.get("previous_index")

    @property
    def args(self) -> t.Mapping[str, Argument]:
        args = self.default_arguments
        args.update(
            {
                name: Argument(
                    name=name,
                    description=arg_info["description"],
                    type=arg_info["type"],
                    default=arg_info["default"] if "default" in arg_info else None,
                    optional=arg_info.get("default") == "None",
                )
                for name, arg_info in self._specification.get("args", {}).items()
            },
        )
        return types.MappingProxyType(args)

    @property
    def specification(self) -> t.Dict[str, t.Any]:
        return copy.deepcopy(self._specification)

    @property
    def default_arguments(self) -> t.Dict[str, Argument]:
        """Add the default arguments of a fondant component."""
        return {
            "input_manifest_path": Argument(
                name="input_manifest_path",
                description="Path to the input manifest",
                type="str",
                optional=True,
            ),
            "component_spec": Argument(
                name="component_spec",
                description="The component specification as a dictionary",
                type="dict",
            ),
            "input_partition_rows": Argument(
                name="input_partition_rows",
                description="The number of rows to load per partition. \
                        Set to override the automatic partitioning",
                type="int",
                optional=True,
            ),
            "cache": Argument(
                name="cache",
                description="Set to False to disable caching, True by default.",
                type="bool",
                default=True,
            ),
            "cluster_type": Argument(
                name="cluster_type",
                description="The cluster type to use for the execution",
                type="str",
                default="default",
            ),
            "client_kwargs": Argument(
                name="client_kwargs",
                description="Keyword arguments to pass to the Dask client",
                type="dict",
                default={},
            ),
            "metadata": Argument(
                name="metadata",
                description="Metadata arguments containing the run id and base path",
                type="str",
            ),
            "output_manifest_path": Argument(
                name="output_manifest_path",
                description="Path to the output manifest",
                type="str",
            ),
            "consumes": Argument(
                name="consumes",
                description="A mapping to update the fields consumed by the operation as defined "
                "in the component spec. The keys are the names of the fields to be "
                "received by the component, while the values are the type of the "
                "field, or the name of the field to map from the input dataset.",
                type="dict",
                default={},
            ),
            "produces": Argument(
                name="produces",
                description="A mapping to update the fields produced by the operation as defined "
                "in the component spec. The keys are the names of the fields to be "
                "produced by the component, while the values are the type of the "
                "field, or the name that should be used to write the field to the "
                "output dataset.",
                type="dict",
                default={},
            ),
        }

    @property
    def kubeflow_specification(self) -> "KubeflowComponentSpec":
        return KubeflowComponentSpec.from_fondant_component_spec(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"

    def __eq__(self, other):
        if not isinstance(other, ComponentSpec):
            return False
        return self._specification == other._specification


class OperationSpec:
    """A spec for the operation, which contains the `consumes` and `produces` sections of the
    component spec, updated with the `consumes` and `produces` mappings provided as arguments to
    the operation.
    """

    def __init__(
        self,
        specification: ComponentSpec,
        *,
        consumes: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        produces: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
    ) -> None:
        self.specification = specification

        self._mappings = {
            "consumes": consumes,
            "produces": produces,
        }
        self._validate_mappings()

        self._inner_consumes: t.Optional[t.Mapping[str, Field]] = None
        self._outer_consumes: t.Optional[t.Mapping[str, Field]] = None
        self._inner_produces: t.Optional[t.Mapping[str, Field]] = None
        self._outer_produces: t.Optional[t.Mapping[str, Field]] = None

    def _validate_mappings(self) -> None:
        """Validate received consumes and produces mappings on their types."""
        for name, mapping in self._mappings.items():
            if not mapping:
                continue
            for key, value in mapping.items():
                if not isinstance(value, (str, pa.DataType)):
                    msg = f"Unexpected type {type(value)} received for key {key} in {name} mapping"
                    raise InvalidPipelineDefinition(msg)

    def _inner_mapping(self, name: str) -> t.Mapping[str, Field]:
        """Calculate the "inner mapping" of the operation. This is the mapping that the component
        `transform` (or equivalent) method will receive. This is calculated by starting from the
        component spec section, and updating it with any string to type mappings from the
        argument mapping.

        Args:
            name: "consumes" or "produces"
        """
        spec_mapping = getattr(self.specification, name)
        args_mapping = self._mappings[name]

        if not args_mapping:
            return spec_mapping

        mapping = dict(spec_mapping)

        for key, value in args_mapping.items():
            if not isinstance(value, pa.DataType):
                continue

            if not self.specification.is_generic(name):
                msg = (
                    f"Component {self.specification.name} does not allow specifying additional "
                    f"fields but received {key}."
                )
                raise InvalidPipelineDefinition(msg)

            if key not in spec_mapping:
                mapping[key] = Field(name=key, type=Type(value))
            else:
                spec_type = spec_mapping[key].type.value
                if spec_type == value:
                    # Same info in mapping and component spec, let it pass
                    pass
                else:
                    msg = (
                        f"Received pyarrow DataType value {value} for key {key} in the "
                        f"`{name}` argument passed to the operation, but {key} is "
                        f"already defined in the `{name}` section of the component spec "
                        f"with type {spec_type}"
                    )
                    raise InvalidPipelineDefinition(msg)

        return types.MappingProxyType(mapping)

    def _outer_mapping(self, name: str) -> t.Mapping[str, Field]:
        """Calculate the "outer mapping" of the operation. This is the mapping that the dataIO
        needs to read / write. This is calculated by starting from the "inner mapping" updating it
        with any string to string mappings from the argument mapping.

        Args:
            name: "consumes" or "produces"
        """
        spec_mapping = getattr(self, f"inner_{name}")
        args_mapping = self._mappings[name]

        if not args_mapping:
            return spec_mapping

        mapping = dict(spec_mapping)

        for key, value in args_mapping.items():
            if not isinstance(value, str):
                continue

            if key in spec_mapping:
                mapping[value] = Field(name=value, type=mapping.pop(key).type)
            else:
                msg = (
                    f"Received a string value for key {key} in the `{name}` "
                    f"argument passed to the operation, but {key} is not defined in "
                    f"the `{name}` section of the component spec."
                )
                raise InvalidPipelineDefinition(msg)

        return types.MappingProxyType(mapping)

    @property
    def inner_consumes(self) -> t.Mapping[str, Field]:
        """The "inner" `consumes` mapping which the component `transform` (or equivalent) method
        will receive.
        """
        if self._inner_consumes is None:
            self._inner_consumes = self._inner_mapping("consumes")

        return self._inner_consumes

    @property
    def outer_consumes(self) -> t.Mapping[str, Field]:
        """The "outer" `consumes` mapping which the dataIO needs to read / write."""
        if self._outer_consumes is None:
            self._outer_consumes = self._outer_mapping("consumes")

        return self._outer_consumes

    @property
    def inner_produces(self) -> t.Mapping[str, Field]:
        """The "inner" `produces` mapping which the component `transform` (or equivalent) method
        will receive.
        """
        if self._inner_produces is None:
            self._inner_produces = self._inner_mapping("produces")

        return self._inner_produces

    @property
    def outer_produces(self) -> t.Mapping[str, Field]:
        """The "outer" `produces` mapping which the dataIO needs to read / write."""
        if self._outer_produces is None:
            self._outer_produces = self._outer_mapping("produces")

        return self._outer_produces


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
    def from_fondant_component_spec(cls, fondant_component: ComponentSpec):
        """Generate a Kubeflow component spec from a Fondant component spec."""
        input_definitions = {
            "parameters": {
                **cls.convert_arguments(fondant_component),
            },
        }

        cleaned_component_name = fondant_component.sanitized_component_name

        specification = {
            "components": {
                "comp-"
                + cleaned_component_name: {
                    "executorLabel": "exec-" + cleaned_component_name,
                    "inputDefinitions": input_definitions,
                },
            },
            "deploymentSpec": {
                "executors": {
                    "exec-"
                    + cleaned_component_name: {
                        "container": {
                            "command": ["fondant", "execute", "main"],
                            "image": fondant_component.image,
                        },
                    },
                },
            },
            "pipelineInfo": {"name": cleaned_component_name},
            "root": {
                "dag": {
                    "tasks": {
                        cleaned_component_name: {
                            "cachingOptions": {"enableCache": True},
                            "componentRef": {"name": "comp-" + cleaned_component_name},
                            "inputs": {
                                "parameters": {
                                    param: {"componentInputParameter": param}
                                    for param in input_definitions["parameters"]
                                },
                            },
                            "taskInfo": {"name": cleaned_component_name},
                        },
                    },
                },
                "inputDefinitions": input_definitions,
            },
            "schemaVersion": "2.1.0",
            "sdkVersion": "kfp-2.0.1",
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
