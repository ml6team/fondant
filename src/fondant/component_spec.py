"""This module defines classes to represent an Fondant component specification."""
import ast
import copy
import json
import pkgutil
import types
import typing as t
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jsonschema.exceptions
import yaml
from jsonschema import Draft4Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

from fondant.exceptions import InvalidComponentSpec
from fondant.schema import Field, KubeflowCommandArguments, Type

# TODO: remove after upgrading to kfpv2
kubeflow_to_python_type_dict = {
    "String": str,
    "Integer": int,
    "Float": float,
    "Boolean": ast.literal_eval,
    "JsonObject": json.loads,
    "JsonArray": json.loads,
}


def kubeflow2python_type(type_: str) -> t.Any:
    map_fn = kubeflow_to_python_type_dict[type_]
    return lambda value: map_fn(value) if value != "None" else None  # type: ignore


# TODO: Change after upgrading to kfp v2
# :https://www.kubeflow.org/docs/components/pipelines/v2/data-types/parameters/
python2kubeflow_type = {
    "str": "String",
    "int": "Integer",
    "float": "Float",
    "bool": "Boolean",
    "dict": "JsonObject",
    "list": "JsonArray",
}


@dataclass
class Argument:
    """
    Kubeflow component argument.

    Args:
        name: name of the argument
        description: argument description
        type: the python argument type (str, int, ...)
        placeholder_type: the kubeflow placeholder type (inputValue, inputPath, ...)
        default: default value of the argument (defaults to None)
    """

    name: str
    description: str
    type: str
    placeholder_type: str = "InputValue"
    default: t.Optional[str] = None


input_manifest_path = Argument(
    name="input_manifest_path",
    description="Path to the input manifest",
    type="str",
    placeholder_type="inputPath",
)

output_manifest_path = Argument(
    name="output_manifest_path",
    description="Path to the output manifest",
    type="str",
    placeholder_type="outputPath",
)

metadata = Argument(
    name="metadata",
    description="Metadata arguments containing the run id and base path",
    type="dict",
    placeholder_type="inputValue",
)

component_spec = Argument(
    name="component_spec",
    description="The component specification as a dictionary",
    type="dict",
    placeholder_type="inputValue",
)


class ComponentType(Enum):
    """Enum representing component base arguments for the different component type."""

    LOAD = "load"
    TRANSFORM = "transform"
    WRITE = "write"

    def get_base_args(self) -> t.Dict[str, Argument]:
        if self == ComponentType.LOAD:
            base_args = {
                "component_spec": component_spec,
                "metadata": metadata,
                "output_manifest_path": output_manifest_path,
            }
        elif self == ComponentType.TRANSFORM:
            base_args = {
                "component_spec": component_spec,
                "metadata": metadata,
                "input_manifest_path": input_manifest_path,
                "output_manifest_path": output_manifest_path,
            }
        elif self == ComponentType.WRITE:
            base_args = {
                "component_spec": component_spec,
                "metadata": metadata,
                "input_manifest_path": input_manifest_path,
            }
        else:
            msg = "Invalid component type."
            raise ValueError(msg)
        return base_args


class ComponentSubset:
    """
    Class representing a Fondant Component subset.

    Args:
        specification: the part of the component json representing the subset
    """

    def __init__(self, specification: t.Dict[str, t.Any]) -> None:
        self._specification = specification

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"

    @property
    def fields(self) -> t.Mapping[str, Field]:
        return types.MappingProxyType(
            {
                name: Field(name=name, type=Type.from_json(field))
                for name, field in self._specification["fields"].items()
            },
        )

    @property
    def additional_fields(self) -> bool:
        return self._specification.get("additionalFields", True)


class ComponentSpec:
    """
    Class representing a Fondant component specification.

    Args:
        specification: The fondant component specification as a Python dict
    """

    def __init__(self, specification: t.Dict[str, t.Any]) -> None:
        self._specification = copy.deepcopy(specification)
        self._validate_spec()
        self._type: ComponentType = self._get_component_type()

    def _get_component_type(self) -> "ComponentType":
        """Function that returns the component type based on the component specification."""
        consumes = self._specification.get("consumes", False)
        produces = self._specification.get("produces", False)

        if not consumes and produces:
            _type = ComponentType.LOAD
        elif consumes and produces:
            _type = ComponentType.TRANSFORM
        elif consumes and not produces:
            _type = ComponentType.WRITE
        else:
            msg = "either 'consumes' or 'produces' fields have to defined for the component"
            raise InvalidComponentSpec(msg)
        return _type

    def _validate_spec(self) -> None:
        """Validate a component specification against the component schema.

        Raises: InvalidComponent when the component specification is not valid.
        """
        spec_data = pkgutil.get_data("fondant", "schemas/component_spec.json")

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

        registry = Registry(retrieve=retrieve_from_filesystem)
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
    def description(self):
        return self._specification["description"]

    @property
    def image(self):
        return self._specification["image"]

    @property
    def index(self):
        return ComponentSubset({"fields": {}})

    @property
    def consumes(self) -> t.Mapping[str, ComponentSubset]:
        """The subsets consumed by the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: ComponentSubset(subset)
                for name, subset in self._specification.get("consumes", {}).items()
                if name != "additionalSubsets"
            },
        )

    @property
    def produces(self) -> t.Mapping[str, ComponentSubset]:
        """The subsets produced by the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: ComponentSubset(subset)
                for name, subset in self._specification.get("produces", {}).items()
                if name != "additionalSubsets"
            },
        )

    @property
    def accepts_additional_subsets(self) -> bool:
        return self._specification.get("consumes", {}).get("additionalSubsets", True)

    @property
    def outputs_additional_subsets(self) -> bool:
        return self._specification.get("produces", {}).get("additionalSubsets", True)

    @property
    def args(self) -> t.Dict[str, Argument]:
        component_base_args = self._type.get_base_args()

        user_args = {
            name: Argument(
                name=name,
                description=arg_info["description"],
                type=arg_info["type"],
                placeholder_type="inputValue",
                default=arg_info["default"] if "default" in arg_info else None,
            )
            for name, arg_info in self._specification.get("args", {}).items()
        }

        return {**component_base_args, **user_args}

    @property
    def specification(self) -> t.Dict[str, t.Any]:
        return copy.deepcopy(self._specification)

    @property
    def type(self) -> ComponentType:
        return self._type

    @property
    def kubeflow_specification(self) -> "KubeflowComponentSpec":
        return KubeflowComponentSpec.from_fondant_component_spec(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"

    def __eq__(self, other):
        if not isinstance(other, ComponentSpec):
            return False
        return self._specification == other._specification


class KubeflowComponentSpec:
    """
    Class representing a Kubeflow component specification.

    Args:
        specification: The kubeflow component specification as a Python dict
    """

    def __init__(self, specification: t.Dict[str, t.Any]) -> None:
        self._specification = specification

    @classmethod
    def from_fondant_component_spec(
        cls,
        fondant_component: ComponentSpec,
    ) -> "KubeflowComponentSpec":
        """Create a Kubeflow component spec from a Fondant component spec."""

        def filter_arguments(args, placeholder_types):
            return [
                {
                    "name": arg.name,
                    "description": arg.description,
                    "type": python2kubeflow_type[arg.type],
                    **({"default": arg.default} if arg.default is not None else {}),
                }
                for arg in args.values()
                if arg.placeholder_type in placeholder_types
            ]

        input_arguments = filter_arguments(
            fondant_component.args,
            ["inputValue", "inputPath"],
        )
        output_arguments = filter_arguments(fondant_component.args, ["outputPath"])

        specification = {
            "name": fondant_component.name,
            "description": fondant_component.description,
            **({"inputs": input_arguments} if input_arguments else {}),
            **({"outputs": output_arguments} if output_arguments else {}),
            "implementation": {
                "container": {
                    "image": fondant_component.image,
                    "command": [
                        "python3",
                        "main.py",
                        *cls._dump_args(fondant_component.args.values()),
                    ],
                },
            },
        }
        return cls(specification)

    @staticmethod
    def _dump_args(args: t.Iterable[Argument]) -> KubeflowCommandArguments:
        """Dump Fondant specification arguments to kfp command arguments."""
        dumped_args: KubeflowCommandArguments = []
        for arg in args:
            arg_name = arg.name.strip().replace(" ", "_")
            arg_name_cmd = f"--{arg_name}"

            dumped_args.append(arg_name_cmd)
            dumped_args.append({arg.placeholder_type: arg_name})

        return dumped_args

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

    @property
    def input_arguments(self) -> t.Mapping[str, Argument]:
        """The input arguments of the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                info["name"]: Argument(
                    name=info["name"],
                    description=info["description"],
                    type=info["type"],
                    default=info["default"] if "default" in info else None,
                )
                for info in self._specification["inputs"]
            },
        )

    @property
    def output_arguments(self) -> t.Mapping[str, Argument]:
        """The output arguments of the component as an immutable mapping."""
        outputs = self._specification.get("outputs")
        return types.MappingProxyType(
            {
                info["name"]: Argument(
                    name=info["name"],
                    description=info["description"],
                    type=info["type"],
                )
                for info in outputs or []
            },
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"
