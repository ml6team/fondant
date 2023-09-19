"""This module defines classes to represent an Fondant component specification."""
import ast
import copy
import json
import pkgutil
import types
import typing as t
from dataclasses import dataclass
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
        default: default value of the argument (defaults to None)
    """

    name: str
    description: str
    type: str
    default: t.Optional[str] = None


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
        return {
            name: Argument(
                name=name,
                description=arg_info["description"],
                type=arg_info["type"],
                default=arg_info["default"] if "default" in arg_info else None,
            )
            for name, arg_info in self._specification.get("args", {}).items()
        }

    @property
    def specification(self) -> t.Dict[str, t.Any]:
        return copy.deepcopy(self._specification)

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
        specification = {
            "name": fondant_component.name,
            "description": fondant_component.description,
            "inputs": [
                {
                    "name": "input_manifest_path",
                    "description": "Path to the input manifest",
                    "type": "String",
                },
                {
                    "name": "metadata",
                    "description": "Metadata arguments containing the run id and base path",
                    "type": "String",
                },
                {
                    "name": "component_spec",
                    "description": "The component specification as a dictionary",
                    "type": "JsonObject",
                    "default": "None",
                },
                {
                    "name": "input_partition_rows",
                    "description": "The number of rows to load per partition. Set to override the"
                    " automatic partitioning",
                    "type": "String",
                    "default": "None",
                },
                {
                    "name": "cache",
                    "description": "Set to False to disable caching, True by default.",
                    "type": "Boolean",
                    "default": "True",
                },
                {
                    "name": "cluster_type",
                    "description": "The type of cluster to use for distributed execution",
                    "type": "String",
                    "default": "default",
                },
                {
                    "name": "client_kwargs",
                    "description": "Keyword arguments used to initialise the dask client",
                    "type": "JsonObject",
                    "default": "{}",
                },
                *(
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "type": python2kubeflow_type[arg.type],
                        **({"default": arg.default} if arg.default is not None else {}),
                    }
                    for arg in fondant_component.args.values()
                ),
            ],
            "outputs": [
                {
                    "name": "output_manifest_path",
                    "description": "Path to the output manifest",
                    "type": "String",
                },
            ],
            "implementation": {
                "container": {
                    "image": fondant_component.image,
                    "command": [
                        "fondant",
                        "execute",
                        "main",
                        "--input_manifest_path",
                        {"inputPath": "input_manifest_path"},
                        "--metadata",
                        {"inputValue": "metadata"},
                        "--component_spec",
                        {"inputValue": "component_spec"},
                        "--input_partition_rows",
                        {"inputValue": "input_partition_rows"},
                        "--cache",
                        {"inputValue": "cache"},
                        *cls._dump_args(fondant_component.args.values()),
                        "--output_manifest_path",
                        {"outputPath": "output_manifest_path"},
                        "--cluster_type",
                        {"inputValue": "cluster_type"},
                        "--client_kwargs",
                        {"inputValue": "client_kwargs"},
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
            dumped_args.append({"inputValue": arg_name})

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
        return types.MappingProxyType(
            {
                info["name"]: Argument(
                    name=info["name"],
                    description=info["description"],
                    type=info["type"],
                )
                for info in self._specification["outputs"]
            },
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"
