"""This module defines classes to represent an Fondant component specification."""
import copy
import json
import pkgutil
import pydoc
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
    type: t.Type
    description: t.Optional[str] = None
    default: t.Optional[t.Any] = None
    optional: t.Optional[bool] = False

    def __post_init__(self):
        self.default = None if self.default == "None" else self.default
        self.parser = json.loads if self.type in [dict, list] else self.type

    @property
    def kubeflow_type(self) -> str:
        lookup = {
            str: "STRING",
            int: "NUMBER_INTEGER",
            float: "NUMBER_DOUBLE",
            bool: "BOOLEAN",
            dict: "STRUCT",
            list: "LIST",
        }
        return lookup[self.type]

    def to_spec(self):
        return {
            k: v
            for k, v in {
                "description": self.description,
                "type": self.type.__name__,
                "default": self.default,
            }.items()
            if v is not None
        }


class ComponentSpec:
    """
    Class representing a Fondant component specification.

    Args:
        name: The name of the component
        image: The docker image uri to use for the component
        description: The description of the component
            consumes: A mapping containing the fields consumed by the operation. The keys are the
            names of the fields to be received by the component, while the values are the
            type of the field.


            produces: A mapping containing the fields produced by the operation. The keys are the
                names of the fields to be produced by the component, while the values are the
                type of the field to be written

            arguments: A dictionary containing the argument name and value for the operation.

    """

    def __init__(
        self,
        name: str,
        image: str,
        *,
        description: t.Optional[str] = None,
        consumes: t.Optional[t.Mapping[str, t.Union[str, pa.DataType, bool]]] = None,
        produces: t.Optional[t.Mapping[str, t.Union[str, pa.DataType, bool]]] = None,
        previous_index: t.Optional[str] = None,
        args: t.Optional[t.Dict[str, t.Any]] = None,
        tags: t.Optional[t.List[str]] = None,
    ):
        spec_dict: t.Dict[str, t.Any] = {
            "name": name,
            "image": image,
        }

        if description:
            spec_dict["description"] = description

        if tags:
            spec_dict["tags"] = tags

        if consumes:
            spec_dict["consumes"] = consumes

        if produces:
            spec_dict["produces"] = produces

        if previous_index:
            spec_dict["previous_index"] = previous_index

        if args:
            spec_dict["args"] = args

        self._specification = spec_dict
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
            return cls.from_dict(specification)

    def to_file(self, path) -> None:
        """Dump the component spec to the file specified by the provided path."""
        with open(path, "w", encoding="utf-8") as file_:
            yaml.dump(self._specification, file_)

    @classmethod
    def from_dict(cls, component_spec_dict: t.Dict[str, t.Any]) -> "ComponentSpec":
        """Load the component spec from a dictionary."""
        try:
            return cls(**component_spec_dict)
        except TypeError as e:
            msg = f"Invalid component spec: {e}"
            raise InvalidComponentSpec(msg)

    @property
    def name(self):
        return self._specification["name"]

    @property
    def safe_name(self):
        return self.sanitized_component_name(self._specification["name"])

    def sanitized_component_name(self, name) -> str:
        """Cleans and converts a component name."""
        return name.lower().replace(" ", "_")

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
                name: Field(name=name, type=Type.from_dict(field))
                for name, field in self._specification.get("consumes", {}).items()
                if name != "additionalProperties"
            },
        )

    @property
    def produces(self) -> t.Mapping[str, Field]:
        """The fields produced by the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(name=name, type=Type.from_dict(field))
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
                    description=arg_info.get("description"),
                    type=pydoc.locate(arg_info["type"]),  # type: ignore
                    default=arg_info.get("default"),
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
                type=str,
                optional=True,
            ),
            "operation_spec": Argument(
                name="operation_spec",
                description="The operation specification as a dictionary",
                type=str,
            ),
            "input_partition_rows": Argument(
                name="input_partition_rows",
                description="The number of rows to load per partition. \
                        Set to override the automatic partitioning",
                type=int,
                optional=True,
            ),
            "cache": Argument(
                name="cache",
                description="Set to False to disable caching, True by default.",
                type=bool,
                default=True,
            ),
            "cluster_type": Argument(
                name="cluster_type",
                description="The cluster type to use for the execution",
                type=str,
                default="default",
            ),
            "client_kwargs": Argument(
                name="client_kwargs",
                description="Keyword arguments to pass to the Dask client",
                type=dict,
                default={},
            ),
            "metadata": Argument(
                name="metadata",
                description="Metadata arguments containing the run id and base path",
                type=str,
            ),
            "output_manifest_path": Argument(
                name="output_manifest_path",
                description="Path to the output manifest",
                type=str,
            ),
        }

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
        component_spec: ComponentSpec,
        *,
        consumes: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
        produces: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]] = None,
    ) -> None:
        self._component_spec = component_spec

        self._mappings = {
            "consumes": consumes,
            "produces": produces,
        }
        self._validate_mappings()

        self._inner_consumes: t.Optional[t.Mapping[str, Field]] = None
        self._outer_consumes: t.Optional[t.Mapping[str, Field]] = None
        self._inner_produces: t.Optional[t.Mapping[str, Field]] = None
        self._outer_produces: t.Optional[t.Mapping[str, Field]] = None

    def to_dict(self) -> dict:
        def _dump_mapping(
            mapping: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]],
        ) -> dict:
            if mapping is None:
                return {}

            serialized_mapping: t.Dict[str, t.Any] = mapping.copy()
            for key, value in mapping.items():
                if isinstance(value, pa.DataType):
                    serialized_mapping[key] = Type(value).to_dict()
            return serialized_mapping

        return {
            "specification": self._component_spec.specification,
            "consumes": _dump_mapping(self._mappings["consumes"]),
            "produces": _dump_mapping(self._mappings["produces"]),
        }

    def to_json(self) -> str:
        specification_dict = self.to_dict()
        return json.dumps(specification_dict)

    @classmethod
    def from_dict(cls, operation_spec_dict: t.Dict[str, t.Any]) -> "OperationSpec":
        def _parse_mapping(
            json_mapping: dict,
        ) -> t.Optional[t.Dict[str, t.Union[str, pa.DataType]]]:
            """Parse a json mapping to a Python mapping with Fondant types."""
            for key, value in json_mapping.items():
                if isinstance(value, dict):
                    json_mapping[key] = Type.from_dict(value).value
            return json_mapping

        return cls(
            component_spec=ComponentSpec.from_dict(
                operation_spec_dict["specification"],
            ),
            consumes=_parse_mapping(operation_spec_dict["consumes"]),
            produces=_parse_mapping(operation_spec_dict["produces"]),
        )

    @classmethod
    def from_json(cls, operation_spec_json: str) -> "OperationSpec":
        return cls.from_dict(json.loads(operation_spec_json))

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
        spec_mapping = getattr(self._component_spec, name)
        args_mapping = self._mappings[name]

        if not args_mapping:
            return spec_mapping

        mapping = dict(spec_mapping)

        for key, value in args_mapping.items():
            if not isinstance(value, pa.DataType):
                continue

            if not self._component_spec.is_generic(name):
                msg = (
                    f"Component {self._component_spec.name} does not allow specifying additional "
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
                        f"`{name}` argument passed to the operation, but `{key}` is "
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
                    f"Received a string value for key `{key}` in the `{name}` "
                    f"argument passed to the operation, but `{key}` is not defined in "
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

    @property
    def component_name(self) -> str:
        """Get the component name."""
        return self._component_spec.safe_name

    @property
    def previous_index(self) -> t.Optional[str]:
        """The name of the index column of the previous component."""
        return self._component_spec.previous_index

    @property
    def args(self) -> t.Mapping[str, Argument]:
        """The component arguments."""
        return self._component_spec.args

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OperationSpec):
            return False

        # Compare component_spec attribute
        if self._component_spec != other._component_spec:
            return False

        # Compare mappings attribute
        if self._mappings != other._mappings:
            return False

        return True
