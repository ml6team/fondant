"""This module defines classes to represent an Express component specification."""
import copy
import json
import yaml
import pkgutil
import types
import typing as t
from dataclasses import dataclass, asdict

import jsonschema.exceptions
from jsonschema import Draft4Validator

from express.exceptions import InvalidComponentSpec
from express.io_utils import load_yaml
from express.common import Field
from express.pipeline_utils import get_kubeflow_type


@dataclass
class Input:
    """
    Component input arguments
    Attributes:
        name: name of the argument
        description: argument description
        type: the python argument type (str, int, ...)
    """

    name: str
    description: str
    type: str


@dataclass
class Output:
    """
    Component output argument
    Attributes:
        name: name of the argument
        description: argument description
    """

    name: str
    description: str


T = t.TypeVar("T", Input, Output)


@dataclass
class KFPComponent:
    """
    A class representing a Kubeflow Pipelines component.
    Attributes:
        name: The name of the component.
        description: The description of the component.
        image: The Docker image for the component.
        args: A dictionary of component arguments.
        inputs: The input parameters for the component.
        outputs: The output parameters for the component.
        command: The command to run the component.
    """

    name: str
    description: str
    image: str
    args: t.Dict[str, t.Dict[str, str]]
    inputs: t.Union[Input, t.List[Input]] = None
    outputs: t.Union[Output, t.List[Output]] = None
    command: t.List[t.Union[str, t.Dict[str, str]]] = None

    def __post_init__(self):
        self.inputs = self._add_defaults_values(
            self.inputs, self._get_default_inputs(), Input
        )

        self.outputs = self._add_defaults_values(
            self.outputs, self._get_default_outputs(), Output
        )

        self._add_input_args()
        self._set_component_run_cmd()

    @staticmethod
    def _get_default_inputs() -> t.List[Input]:
        """
        Set default inputs to components
        """
        inputs = [
            Input(
                name="input_manifest",
                description="Path to the the input manifest",
                type="String",
            ),
            Input(
                name="extra_args",
                description="The extra arguments passed to the component",
                type="String",
            ),
        ]
        return inputs

    @staticmethod
    def _get_default_outputs() -> t.List[Output]:
        """
        Set default output to components
        """
        outputs = [
            Output(
                name="output_manifest", description="The path to the output manifest"
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
                Input(
                    name=arg_name.strip(),
                    description=arg_info["description"].strip(),
                    type=get_kubeflow_type(arg_info["type"].strip()),
                )
            )

    def _set_component_run_cmd(self):
        """
        Function that return the run command of the kubeflow component
        Returns:
            The kubeflow component run command
        """

        def _add_run_arguments(args: t.List[T]):
            for arg in args:
                arg_name = arg.name.replace("-", "_").strip()
                arg_name_cmd = f'--{arg.name.replace("_", "-")}'.strip()

                if arg_name == "input_manifest":
                    arg_kfp_value_type = "inputPath"
                elif arg_name == "output_manifest":
                    arg_kfp_value_type = "outputPath"
                else:
                    arg_kfp_value_type = "inputValue"

                self.command.append(arg_name_cmd)
                self.command.append({arg_kfp_value_type: arg_name})

        self.command = ["python3", "main.py"]
        _add_run_arguments(self.inputs)
        _add_run_arguments(self.outputs)

    def get_specification(self) -> dict:
        """
        Function that returns the specifications of the kubeflow component
        Returns:
            the kubeflow specification as a dictionary
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
    def fields(self) -> t.Dict[str, Field]:
        return {
            name: Field(name=name, type=field["type"])
            for name, field in self._specification["fields"].items()
        }


class ExpressComponent:
    """
    Class representing an Express component
    Args:
        yaml_spec_path: The yaml file containing the component specification
    """

    def __init__(self, yaml_spec_path: str):
        self.yaml_spec = load_yaml(yaml_spec_path)
        self._kubeflow_comp_specs = self.get_kubeflow_comp_specification()
        self._specification = self.get_express_comp_specification()
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate a component specification against the component schema
        Raises: InvalidManifest when the manifest is not valid.
        """
        spec_schema = json.loads(
            pkgutil.get_data("express", "schemas/component_spec.json")
        )
        validator = Draft4Validator(spec_schema)
        try:
            validator.validate(self._specification)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidComponentSpec.create_from(e)

    def get_kubeflow_comp_specification(self) -> t.Dict[str, any]:
        """
        Function that returns the kubeflow specifications as a dictionary
        """

        kfp_component = KFPComponent(
            name=self.yaml_spec["name"],
            description=self.yaml_spec["description"],
            image=self.yaml_spec["image"],
            args=self.yaml_spec["args"],
        )

        return kfp_component.get_specification()

    def get_express_comp_specification(self) -> t.Dict[str, any]:
        """
        Function that return the express component specification which contains both the kubeflow
        component specifications in addition to the subsets defining the input and output datasets
        """

        express_component_spec = copy.deepcopy(self._kubeflow_comp_specs)
        express_component_spec["input_subsets"] = self.yaml_spec["input_subsets"]
        express_component_spec["output_subsets"] = self.yaml_spec["output_subsets"]

        return express_component_spec

    def get_subset(self, subset_field: str) -> t.Mapping[str, ComponentSubset]:
        """Function that returns subsets from a component specification"""
        return types.MappingProxyType(
            {
                name: ComponentSubset(subset)
                for name, subset in self._specification[subset_field].items()
            }
        )

    def write_component(self, path: str):
        """
        Function that write the component yaml file required to compile a kubeflow pipeline
        """
        with open(path, "w") as file:
            yaml.dump(
                self._kubeflow_comp_specs,
                file,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

    @property
    def input_subsets(self) -> t.Mapping[str, ComponentSubset]:
        return self.get_subset("input_subsets")

    @property
    def output_subsets(self) -> t.Mapping[str, ComponentSubset]:
        return self.get_subset("output_subsets")

    @property
    def name(self):
        return self._specification["name"]

    @property
    def description(self):
        return self._specification["description"]

    @property
    def image(self):
        return self._specification["implementation"]["container"]["image"]

    @property
    def run_command(self):
        return self._specification["implementation"]["container"]["command"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r}"
