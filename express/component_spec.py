"""This module defines classes to represent an Express component specification."""
import copy
import json
import yaml
import pkgutil
import types
import typing as t
from pathlib import Path
from dataclasses import dataclass, asdict

import jsonschema.exceptions
from jsonschema import Draft4Validator
from jsonschema.validators import RefResolver

from express.exceptions import InvalidComponentSpec
from express.schema import Field


def get_kubeflow_type(python_type: str):
    """
    Function that returns a Kubeflow equivalent data type from a Python data type
    Args:
        python_type (str): the string representation of the data type
    Returns:
        The kubeflow data type
    """
    mapping = {
        "str": "String",
        "int": "Integer",
        "float": "Float",
        "bool": "Boolean",
        "dict": "Map",
        "list": "List",
        "tuple": "List",
        "set": "Set",
    }

    try:
        return mapping[python_type]
    except KeyError:
        raise ValueError(f"Invalid Python data type: {python_type}")


@dataclass
class KubeflowInput:
    """
    Kubeflow component input argument
    Args:
        name: name of the argument
        description: argument description
        type: the python argument type (str, int, ...)
    """

    name: str
    description: str
    type: str


@dataclass
class KubeflowOutput:
    """
    Kubeflow component output argument
    Args:
        name: name of the argument
        description: argument description
    """

    name: str
    description: str


@dataclass
class KubeflowComponent:
    """
    A class representing a Kubeflow Pipelines component.
    Args:
        name: The name of the component.
        description: The description of the component.
        image: The Docker image url for the component.
        inputs: The input parameters for the component.
        outputs: The output parameters for the component.
        command: The command to run the component.
    """

    name: str
    description: str
    image: str
    inputs: t.Union[KubeflowInput, t.List[KubeflowInput]]
    outputs: t.Union[KubeflowOutput, t.List[KubeflowOutput]]
    command: t.List[t.Union[str, t.Dict[str, str]]] = None

    def __post_init__(self):
        self._set_component_run_command()

    def _set_component_run_command(self):
        """
        Function that returns the run command of the Kubeflow component
        Returns:
            The Kubeflow component run command
        """

        def _add_run_arguments(args: t.List[t.Union[KubeflowInput, KubeflowOutput]]):
            for arg in args:
                arg_name = arg.name.replace("-", "_").strip()
                arg_name_cmd = f'--{arg.name.replace("_", "-")}'.strip()

                if arg_name == "input_manifest_path":
                    arg_value_type = "inputPath"
                elif arg_name == "output_manifest_path":
                    arg_value_type = "outputPath"
                else:
                    arg_value_type = "inputValue"

                self.command.append(arg_name_cmd)
                self.command.append({arg_value_type: arg_name})

        self.command = ["python3", "main.py"]
        _add_run_arguments(self.inputs)
        _add_run_arguments(self.outputs)

    @property
    def specification(self) -> dict:
        """
        Function that returns the specification of the kubeflow component
        Returns:
            The Kubeflow specification as a dictionary
        """
        if not all(
                [
                    self.name,
                    self.description,
                    self.inputs,
                    self.outputs,
                    self.image,
                    self.command,
                ]
        ):
            raise ValueError("Missing required attributes to construct specification")

        specification = {
            "name": self.name,
            "description": self.description,
            "inputs": [asdict(input_obj) for input_obj in self.inputs],
            "outputs": [asdict(output_obj) for output_obj in self.outputs],
            "implementation": {
                "container": {"image": self.image, "command": self.command}
            },
        }

        return specification


class ComponentSubset:
    """
    Class representing an Express Component subset.
    """

    def __init__(self, specification: dict) -> None:
        """
        Initialize subsets
        Args:
            specification: the part of the component json representing the subset
        """
        self._specification = specification

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r}"

    @property
    def fields(self) -> t.Mapping[str, Field]:
        return types.MappingProxyType(
            {
                name: Field(name=name, type=field["type"])
                for name, field in self._specification["fields"].items()
            }
        )


class ExpressComponent:
    """
    Class representing an Express component
    Args:
        yaml_spec_path: The yaml file containing the component specification
    """

    def __init__(self, yaml_spec_path: str):
        self.yaml_spec = yaml.safe_load(open(yaml_spec_path, 'r'))
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate a component specification against the component schema
        Raises: InvalidManifest when the manifest is not valid.
        """
        spec_schema = json.loads(
            pkgutil.get_data("express", "schemas/component_spec.json")
        )

        base_uri = (Path(__file__).parent / "schemas").as_uri()
        resolver = RefResolver(base_uri=f"{base_uri}/", referrer=spec_schema)
        validator = Draft4Validator(spec_schema, resolver=resolver)

        try:
            validator.validate(self.yaml_spec)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidComponentSpec.create_from(e)

    def write_kubeflow_component(self, path: str):
        """
        Function that writes the component yaml file required to compile a Kubeflow pipeline
        """
        with open(path, "w") as file:
            yaml.dump(
                self.kubeflow_component_specification,
                file,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    @property
    def kubeflow_component_specification(self) -> t.Dict[str, any]:
        """
        The Kubeflow component specification
        """
        inputs = self.default_input_arguments
        outputs = self.default_output_arguments

        for arg_name, arg_info in self.yaml_spec["args"].items():
            inputs.append(
                KubeflowInput(
                    name=arg_name.strip(),
                    description=arg_info["description"].strip(),
                    type=get_kubeflow_type(arg_info["type"].strip()),
                )
            )

        kubeflow_component = KubeflowComponent(
            name=self.yaml_spec["name"],
            description=self.yaml_spec["description"],
            image=self.yaml_spec["image"],
            inputs=inputs,
            outputs=outputs,
        )

        return kubeflow_component.specification

    @property
    def express_component_specification(self) -> t.Dict[str, any]:
        """
        The express component specification which contains both the Kubeflow component
        specifications in addition to the input and output subsets
        """
        express_component_specification = copy.deepcopy(
            self.kubeflow_component_specification
        )
        express_component_specification["input_subsets"] = self.yaml_spec[
            "input_subsets"
        ]
        express_component_specification["output_subsets"] = self.yaml_spec[
            "output_subsets"
        ]

        return express_component_specification

    @property
    def default_input_arguments(self) -> t.List[KubeflowInput]:
        """The default component input arguments"""
        inputs = [
            KubeflowInput(
                name="input_manifest_path",
                description="Path to the the input manifest",
                type="String",
            )
        ]
        return inputs

    @property
    def default_output_arguments(self) -> t.List[KubeflowOutput]:
        """The default component output arguments"""

        outputs = [
            KubeflowOutput(
                name="output_manifest_path",
                description="The path to the output manifest",
            )
        ]
        return outputs

    @property
    def input_subsets(self) -> t.Mapping[str, ComponentSubset]:
        """The input subsets of the component as an immutable mapping"""
        return types.MappingProxyType(
            {
                name: ComponentSubset(subset)
                for name, subset in self.express_component_specification[
                "input_subsets"
            ].items()
            }
        )

    @property
    def output_subsets(self) -> t.Mapping[str, ComponentSubset]:
        """The output subsets of the component as an immutable mapping"""
        return types.MappingProxyType(
            {
                name: ComponentSubset(subset)
                for name, subset in self.express_component_specification[
                "output_subsets"
            ].items()
            }
        )

    @property
    def name(self):
        return self.express_component_specification["name"]

    @property
    def description(self):
        return self.express_component_specification["description"]

    @property
    def image(self):
        return self.express_component_specification["implementation"]["container"][
            "image"
        ]

    @property
    def run_command(self):
        return self.express_component_specification["implementation"]["container"][
            "command"
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.express_component_specification!r}"
