"""This module defines classes to represent an Express manifest."""

import json
import pkgutil
import typing as t
from dataclasses import dataclass

import jsonschema.exceptions
from jsonschema import Draft4Validator

from express.exceptions import InvalidManifest


@dataclass
class Field:
    """Class representing a single field or column in an Express subset."""

    name: str
    type: str


class Subset:
    """
    Class representing an Express subset.

    Args:
        specification: The part of the manifest json representing the subset
        base_path: The base path which the subset location is defined relative to
    """

    def __init__(self, specification: dict, *, base_path: str) -> None:
        self._specification = specification
        self._base_path = base_path

    @property
    def location(self) -> str:
        """The resolved location of the subset"""
        return self._base_path.rstrip("/") + self._specification["location"]

    @property
    def fields(self) -> t.Dict[str, Field]:
        return {
            name: Field(name=name, type=field["type"])
            for name, field in self._specification["fields"].items()
        }


class Index(Subset):
    """Special case of a subset for the index, which has fixed fields"""

    @property
    def fields(self) -> t.Dict[str, Field]:
        return {
            "id": Field(name="id", type="str"),
            "source": Field(name="source", type="str"),
        }


class Manifest:
    """
    Class representing an Express manifest

    Args:
        specification: The manifest specification as a Python dict
    """

    def __init__(self, specification: dict) -> None:
        self._validate_spec(specification)
        self._specification = specification

    @classmethod
    def _validate_spec(cls, spec: dict) -> None:
        """Validate a manifest specification against the manifest schema

        Raises: InvalidManifest when the manifest is not valid.
        """
        spec_schema = json.loads(pkgutil.get_data("express", "schemas/manifest.json"))
        validator = Draft4Validator(spec_schema)
        try:
            validator.validate(spec)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidManifest.create_from(e)

    @classmethod
    def from_file(cls, path: str) -> "Manifest":
        """Load the manifest from the file specified by the provided path"""
        with open(path, encoding="utf-8") as file_:
            specification = json.load(file_)
            return cls(specification)

    def to_file(self, path) -> None:
        """Dump the manifest to the file specified by the provided path"""
        with open(path, "w", encoding="utf-8") as file_:
            json.dump(self._specification, file_)

    @property
    def metadata(self) -> dict:
        return self._specification.get("metadata")

    @property
    def base_path(self) -> str:
        return self.metadata.get("base_path")

    @property
    def index(self) -> Index:
        return Index(self._specification.get("index"), base_path=self.base_path)

    @property
    def subsets(self) -> t.Dict[str, Subset]:
        return {
            name: Subset(subset, base_path=self.base_path)
            for name, subset in self._specification["subsets"].items()
        }
