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
from express.io_utils import load_yaml
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


T = t.TypeVar("T", KubeflowInput, KubeflowOutput)


@dataclass
class KubeflowComponent:
    """
    A class representing a Kubeflow Pipelines component.
    Args:
        name: The name of the component.
        description: The description of the component.
        image: The Docker image url for the component.
        args: A dictionary of component arguments.
        inputs: The input parameters for the component.
        outputs: The output parameters for the component.
        command: The command to run the component.
    """

    name: str
    description: str
    image: str
    args: t.Dict[str, t.Dict[str, str]]
    inputs: t.Union[KubeflowInput, t.List[KubeflowInput]] = None
    outputs: t.Union[KubeflowOutput, t.List[KubeflowOutput]] = None
    command: t.List[t.Union[str, t.Dict[str, str]]] = None

    def __post_init__(self):
        self.inputs = self._add_defaults_values(
            self.inputs, self._get_default_inputs(), KubeflowInput
        )

        self.outputs = self._add_defaults_values(
            self.outputs, self._get_default_outputs(), KubeflowOutput
        )

        self._add_input_args()
        self._set_component_run_command()

    @staticmethod
    def _get_default_inputs() -> t.List[KubeflowInput]:
        """
        Set default inputs to components
        """
        inputs = [
            KubeflowInput(
                name="input_manifest_path",
                description="Path to the the input manifest",
                type="String",
            ),
            KubeflowInput(
                name="args",
                description="The extra arguments passed to the component",
                type="String",
            ),
        ]
        return inputs

    @staticmethod
    def _get_default_outputs() -> t.List[KubeflowOutput]:
        """
        Set default output to components
        """
        outputs = [
            KubeflowOutput(
                name="output_manifest_path",
                description="The path to the output manifest",
            )
        ]
        return outputs

    @staticmethod
    def _add_defaults_values(
            values: t.Union[None, T, t.List[T]],
            default_values: t.List[T],
            value_type: t.Type[T],
    ) -> t.List[T]:
        """Add default values to a input/output list attribute"""

        if values is None:
            values = []

        if isinstance(values, value_type):
            values = [values]

        values.extend(default_values)
        return values

    def _add_input_args(self):
        """Add specified component arguments to the input"""
        for arg_name, arg_info in self.args.items():
            self.inputs.append(
                KubeflowInput(
                    name=arg_name.strip(),
                    description=arg_info["description"].strip(),
                    type=get_kubeflow_type(arg_info["type"].strip()),
                )
            )

    def _set_component_run_command(self):
        """
        Function that returns the run command of the Kubeflow component
        Returns:
            The Kubeflow component run command
        """

        def _add_run_arguments(args: t.List[T]):
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

    def get_specification(self) -> dict:
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
                name: Field(name=name, type=field)
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
        self.yaml_spec = load_yaml(yaml_spec_path)
        self._validate_spec()
        self._kubeflow_component_specification = (
            self.set_kubeflow_component_specification()
        )
        self._express_component_specification = (
            self.set_express_component_specification()
        )

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

    def set_kubeflow_component_specification(self) -> t.Dict[str, any]:
        """
        Function that returns the Kubeflow specification as a dictionary
        """

        kubeflow_component = KubeflowComponent(
            name=self.yaml_spec["name"],
            description=self.yaml_spec["description"],
            image=self.yaml_spec["image"],
            args=self.yaml_spec["args"],
        )

        return kubeflow_component.get_specification()

    def set_express_component_specification(self) -> t.Dict[str, any]:
        """
        Function that return the express component specification which contains both the Kubeflow
        component specifications in addition to the subsets defining the input and output datasets
        """
        express_component_spec = copy.deepcopy(self._kubeflow_component_specification)
        express_component_spec["input_subsets"] = self.yaml_spec["input_subsets"]
        express_component_spec["output_subsets"] = self.yaml_spec["output_subsets"]

        return express_component_spec

    def write_kubeflow_component(self, path: str):
        """
        Function that writes the component yaml file required to compile a Kubeflow pipeline
        """
        with open(path, "w") as file:
            yaml.dump(
                self._kubeflow_component_specification,
                file,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    @property
    def input_subsets(self) -> t.Mapping[str, ComponentSubset]:
        """Function that returns the input subsets from a component specification"""
        return types.MappingProxyType(
            {
                name: ComponentSubset(subset)
                for name, subset in self._express_component_specification[
                "input_subsets"
            ].items()
            }
        )

    @property
    def output_subsets(self) -> t.Mapping[str, ComponentSubset]:
        """Function that returns the output subsets from a component specification"""
        return types.MappingProxyType(
            {
                name: ComponentSubset(subset)
                for name, subset in self._express_component_specification[
                "output_subsets"
            ].items()
            }
        )

    @property
    def name(self):
        return self._express_component_specification["name"]

    @property
    def description(self):
        return self._express_component_specification["description"]

    @property
    def image(self):
        return self._express_component_specification["implementation"]["container"][
            "image"
        ]

    @property
    def run_command(self):
        return self._express_component_specification["implementation"]["container"][
            "command"
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._express_component_specification!r}"


com = ExpressComponent(
    "/home/philippe/Scripts/express/tests/component_example/valid_component/express_component.yaml")
print(com.yaml_spec)