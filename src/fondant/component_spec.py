"""This module defines classes to represent an Fondant component specification."""
import copy
import json
import pkgutil
import re
import types
import typing as t
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path

import jsonschema.exceptions
import yaml
from jsonschema import Draft4Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

from fondant.exceptions import InvalidComponentSpec
from fondant.schema import Field, KubeflowCommandArguments, Type


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
    default: t.Optional[str] = None
    optional: t.Optional[bool] = False

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
        return lookup[self.type]

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
        def _is_optional(arg_information):
            if "default" in arg_information:
                return arg_information["default"] == "None"
            return False

        return {
            name: Argument(
                name=name,
                description=arg_info["description"],
                type=arg_info["type"],
                default=arg_info["default"] if "default" in arg_info else None,
                optional=_is_optional(arg_info),
            )
            for name, arg_info in self._specification.get("args", {}).items()
        }

    @property
    def specification(self) -> t.Dict[str, t.Any]:
        return copy.deepcopy(self._specification)

    @property
    def input_arguments(self) -> t.Mapping[str, Argument]:
        """The input arguments (default + custom) of the component as an immutable mapping."""
        args = self.args

        # Add default arguments
        args.update(
            {
                "input_manifest_path": Argument(
                    name="input_manifest_path",
                    description="Path to the input manifest",
                    type="str",
                    default=None,
                ),
                "component_spec": Argument(
                    name="component_spec",
                    description="The component specification as a dictionary",
                    type="dict",
                    default={},
                ),
                "input_partition_rows": Argument(
                    name="input_partition_rows",
                    description="The number of rows to load per partition. \
                        Set to override the automatic partitioning",
                    type="str",
                    default=None,
                ),
                "cache": Argument(
                    name="cache",
                    description="Set to False to disable caching, True by default.",
                    type="bool",
                    default=True,
                ),
                "metadata": Argument(
                    name="metadata",
                    description="Metadata arguments containing the run id and base path",
                    type="str",
                    default=None,
                ),
            },
        )
        return types.MappingProxyType(args)

    @property
    def output_arguments(self) -> t.Mapping[str, Argument]:
        """The output arguments of the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                "output_manifest_path": Argument(
                    name="output_manifest_path",
                    description="Path to the output manifest",
                    type="str",
                    default=None,
                ),
            },
        )

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

            if arg.optional and arg.default == "None":
                arg_type_dict["isOptional"] = True
            if arg.default is not None and arg.default != "None":
                arg_type_dict["defaultValue"] = arg.default

            args[arg.name] = {
                "parameterType": arg.kubeflow_type,
                "description": arg.description,
                **arg_type_dict,
            }

        return args

    @staticmethod
    def sanitize_component_name(name: str) -> str:
        """Cleans and converts a name to be kfp V2 compatible.

        Taken from https://github.com/kubeflow/pipelines/blob/
        cfe671c485d4ee8514290ee81ca2785e8bda5c9b/sdk/python/kfp/dsl/utils.py#L52
        """
        return (
            re.sub("-+", "-", re.sub("[^-0-9a-z]+", "-", name.lower()))
            .lstrip("-")
            .rstrip("-")
        )

    @classmethod
    def from_fondant_component_spec(cls, fondant_component: ComponentSpec):
        """Generate a Kubeflow component spec from a ComponentOp."""
        input_definitions = {
            "artifacts": {
                "input_manifest_path": {
                    "description": "Path to the input manifest",
                    "artifactType": {
                        "schemaTitle": "system.Artifact",
                        "schemaVersion": "0.0.1",
                    },
                    "isOptional": True,
                },
            },
            "parameters": {
                "component_spec": {
                    "description": "The component specification as a dictionary",
                    "defaultValue": {},
                    "isOptional": True,
                    "parameterType": "STRUCT",
                },
                "input_partition_rows": {
                    "description": "The number of rows to load per partition."
                    + " Set to override the automatic partitioning",
                    "isOptional": True,
                    "parameterType": "NUMBER_INTEGER",
                },
                "cache": {
                    "parameterType": "BOOLEAN",
                    "description": "Set to False to disable caching, True by default.",
                    "defaultValue": True,
                    "isOptional": True,
                },
                "metadata": {
                    "description": "Metadata arguments containing the run id and base path",
                    "parameterType": "STRING",
                },
                **cls.convert_arguments(fondant_component),
            },
        }

        cleaned_component_name = cls.sanitize_component_name(fondant_component.name)

        output_definitions = {
            "artifacts": {
                "output_manifest_path": {
                    "artifactType": {
                        "schemaTitle": "system.Artifact",
                        "schemaVersion": "0.0.1",
                    },
                    "description": "Path to the output manifest",
                },
            },
        }

        specification = {
            "components": {
                "comp-"
                + cleaned_component_name: {
                    "executorLabel": "exec-" + cleaned_component_name,
                    "inputDefinitions": input_definitions,
                    "outputDefinitions": output_definitions,
                },
            },
            "deploymentSpec": {
                "executors": {
                    "exec-"
                    + cleaned_component_name: {
                        "container": {
                            "args": [
                                "--input_manifest_path",
                                "{{$.inputs.artifacts['input_manifest_path'].uri}}",
                                "--metadata",
                                "{{$.inputs.parameters['metadata']}}",
                                "--component_spec",
                                "{{$.inputs.parameters['component_spec']}}",
                                "--input_partition_rows",
                                "{{$.inputs.parameters['input_partition_rows']}}",
                                "--cache",
                                "{{$.inputs.parameters['cache']}}",
                                *cls._dump_args(fondant_component.args.values()),
                                "--output_manifest_path",
                                "{{$.outputs.artifacts['output_manifest_path'].uri}}",
                            ],
                            "command": ["fondant", "execute", "main"],
                            "image": fondant_component.image,
                        },
                    },
                },
            },
            "pipelineInfo": {"name": cleaned_component_name},
            "root": {
                "dag": {
                    "outputs": {
                        "artifacts": {
                            "output_manifest_path": {
                                "artifactSelectors": [
                                    {
                                        "outputArtifactKey": "output_manifest_path",
                                        "producerSubtask": cleaned_component_name,
                                    },
                                ],
                            },
                        },
                    },
                    "tasks": {
                        cleaned_component_name: {
                            "cachingOptions": {"enableCache": True},
                            "componentRef": {"name": "comp-" + cleaned_component_name},
                            "inputs": {
                                "artifacts": {
                                    "input_manifest_path": {
                                        "componentInputArtifact": "input_manifest_path",
                                    },
                                },
                                "parameters": {
                                    "component_spec": {
                                        "componentInputParameter": "component_spec",
                                    },
                                    "input_partition_rows": {
                                        "componentInputParameter": "input_partition_rows",
                                    },
                                    "metadata": {"componentInputParameter": "metadata"},
                                    "cache": {"componentInputParameter": "cache"},
                                },
                            },
                            "taskInfo": {"name": cleaned_component_name},
                        },
                    },
                },
                "inputDefinitions": input_definitions,
                "outputDefinitions": output_definitions,
            },
            "schemaVersion": "2.1.0",
            "sdkVersion": "kfp-2.0.1",
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
            dumped_args.append("{{$.inputs.parameters['" + f"{arg_name}" + "']}}")

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"
