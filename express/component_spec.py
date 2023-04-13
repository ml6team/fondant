"""This module defines classes to represent an Express component specification."""
import json
import pkgutil
import yaml
import typing as t
from dataclasses import dataclass

import jsonschema.exceptions
from jsonschema import Draft4Validator

from express.exceptions import InvalidComponentSpec


def load_yaml(yaml_path: str) -> t.Dict:
    """Loads a YAML file and returns a dictionary."""
    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load YAML file: {e}")


@dataclass
class ComponentField:
    """Class representing a single field or column in an Express component subset."""

    name: str
    type: str


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

    @property
    def fields(self) -> t.Dict[str, ComponentField]:
        return {
            name: ComponentField(name=name, type=field["type"])
            for name, field in self._specification["fields"].items()
        }


class ExpressComponent:
    """
    Class representing an Express component
    Args:
        specification: The component specification as a Python dict
        yaml_spec_path: the path to the component yaml file containing the name and the descrption
        of the component
    """

    def __init__(self, specification: dict, yaml_spec_path: str) -> None:
        self._validate_spec(specification)
        self._specification = specification
        self._metadata = load_yaml(yaml_spec_path)

    @staticmethod
    def _validate_spec(spec: dict) -> None:
        """Validate a component specification against the component schema
        Raises: InvalidManifest when the manifest is not valid.
        """
        spec_schema = json.loads(
            pkgutil.get_data("express", "schemas/component_spec.json")
        )
        validator = Draft4Validator(spec_schema)
        try:
            validator.validate(spec)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidComponentSpec.create_from(e)

    def get_subset(self, subset_field: str) -> t.Dict[str, ComponentSubset]:
        """Function that returns subsets from a component specification"""
        return {
            name: ComponentSubset(subset)
            for name, subset in self._specification[subset_field].items()
        }

    @property
    def name(self):
        return self._metadata["name"]

    @property
    def description(self):
        return self._metadata["description"]

    @property
    def input_subsets(self) -> t.Dict[str, ComponentSubset]:
        return self.get_subset("input_subsets")

    @property
    def output_subsets(self) -> t.Dict[str, ComponentSubset]:
        return self.get_subset("output_subsets")

    def to_file(self, path) -> None:
        """Dump the manifest to the file specified by the provided path"""
        with open(path, "w", encoding="utf-8") as file_:
            json.dump(self._specification, file_)

    @classmethod
    def from_file(cls, path: str, yaml_path: str) -> "ExpressComponent":
        """Load the manifest from the file specified by the provided path"""
        with open(path, encoding="utf-8") as file_:
            specification = json.load(file_)
            return cls(specification, yaml_path)
