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
import yaml
from jsonschema import Draft4Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

from fondant.core.exceptions import InvalidComponentSpec
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
            },
        )

    @property
    def produces(self) -> t.Mapping[str, Field]:
        """The fields produced by the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(name=name, type=Type.from_json(field))
                for name, field in self._specification.get("produces", {}).items()
            },
        )

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
