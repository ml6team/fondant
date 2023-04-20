"""This module defines classes to represent an Express manifest."""
import copy
import json
import pkgutil
import types
import typing as t
from pathlib import Path

import jsonschema.exceptions
from jsonschema import Draft4Validator
from jsonschema.validators import RefResolver

from express.exceptions import InvalidManifest
from express.schema import Type, Field


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
    def fields(self) -> t.Mapping[str, Field]:
        """The fields of the subset returned as a immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(name=name, type=field["type"])
                for name, field in self._specification["fields"].items()
            }
        )

    def add_field(self, name: str, type_: Type) -> None:
        if name in self._specification["fields"]:
            raise ValueError("A field with name {name} already exists")

        self._specification["fields"][name] = {"type": type_.value}

    def remove_field(self, name: str) -> None:
        del self._specification["fields"][name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r}"


class Index(Subset):
    """Special case of a subset for the index, which has fixed fields"""

    @property
    def fields(self) -> t.Dict[str, Field]:
        return {
            "id": Field(name="id", type=Type.utf8),
            "source": Field(name="source", type=Type.utf8),
        }


class Manifest:
    """
    Class representing an Express manifest

    Args:
        specification: The manifest specification as a Python dict
    """

    def __init__(self, specification: t.Optional[dict] = None) -> None:
        self._specification = copy.deepcopy(specification)
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate a manifest specification against the manifest schema

        Raises: InvalidManifest when the manifest is not valid.
        """
        spec_schema = json.loads(pkgutil.get_data("express", "schemas/manifest.json"))

        base_uri = (Path(__file__).parent / "schemas").as_uri()
        resolver = RefResolver(base_uri=f"{base_uri}/", referrer=spec_schema)
        validator = Draft4Validator(spec_schema, resolver=resolver)

        try:
            validator.validate(self._specification)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidManifest.create_from(e)

    @classmethod
    def create(cls, *, base_path: str, run_id: str, component_id: str) -> "Manifest":
        """Create an empty manifest

        Args:
            base_path: The base path of the manifest
            run_id: The id of the current pipeline run
            component_id: The id of the current component being executed
        """
        specification = {
            "metadata": {
                "base_path": base_path,
                "run_id": run_id,
                "component_id": component_id,
            },
            "index": {"location": f"/custom_artifact/{run_id}/{component_id}/index"},
            "subsets": {},
        }
        return cls(specification)

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

    def copy(self):
        """Return a deep copy of itself"""
        return self.__class__(copy.deepcopy(self._specification))

    @property
    def metadata(self) -> dict:
        return self._specification["metadata"]

    def add_metadata(self, key: str, value: t.Any) -> None:
        self.metadata[key] = value

    @property
    def base_path(self) -> str:
        return self.metadata["base_path"]

    @property
    def run_id(self) -> str:
        return self.metadata["run_id"]

    @property
    def component_id(self) -> str:
        return self.metadata["component_id"]

    @property
    def index(self) -> Index:
        return Index(self._specification["index"], base_path=self.base_path)

    @property
    def subsets(self) -> t.Mapping[str, Subset]:
        """The subsets of the manifest as an immutable mapping"""
        return types.MappingProxyType(
            {
                name: Subset(subset, base_path=self.base_path)
                for name, subset in self._specification["subsets"].items()
            }
        )

    def get_subset(self, name: str) -> Subset:
        if name not in self._specification["subsets"]:
            raise ValueError(f"Subset {name} not found in specification")

        return self._specification["subsets"][name]

    def add_subset(self, name: str, fields: t.List[t.Tuple[str, Type]]) -> None:
        if name in self._specification["subsets"]:
            raise ValueError("A subset with name {name} already exists")

        self._specification["subsets"][name] = {
            "location": f"custom_artifact/{self.run_id}/{self.component_id}/{name}",
            "fields": {name: {"type": type_} for name, type_ in fields},
        }

    def remove_subset(self, name: str) -> None:
        if name not in self._specification["subsets"]:
            raise ValueError(f"Subset {name} not found in specification")

        del self._specification["subsets"][name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r}"
