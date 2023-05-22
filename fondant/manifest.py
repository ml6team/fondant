"""This module defines classes to represent a Fondant manifest."""
import copy
import json
import pkgutil
import types
import typing as t
from pathlib import Path

import jsonschema.exceptions
from jsonschema import Draft4Validator
from jsonschema.validators import RefResolver

from fondant.component_spec import ComponentSpec
from fondant.exceptions import InvalidManifest
from fondant.schema import Field, Type


class Subset:
    """
    Class representing a Fondant subset.

    Args:
        specification: The part of the manifest json representing the subset
        base_path: The base path which the subset location is defined relative to
    """

    def __init__(self, specification: dict, *, base_path: str) -> None:
        self._specification = specification
        self._base_path = base_path

    @property
    def location(self) -> str:
        """The absolute location of the subset."""
        return self._base_path + self._specification["location"]

    @property
    def fields(self) -> t.Mapping[str, Field]:
        """The fields of the subset returned as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(name=name, type=Type[field["type"]])
                for name, field in self._specification["fields"].items()
            }
        )

    def add_field(self, name: str, type_: Type, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._specification["fields"]:
            raise ValueError(f"A field with name {name} already exists")

        self._specification["fields"][name] = {"type": type_.name}

    def remove_field(self, name: str) -> None:
        del self._specification["fields"][name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"


class Index(Subset):
    """Special case of a subset for the index, which has fixed fields."""

    @property
    def fields(self) -> t.Dict[str, Field]:
        return {
            "id": Field(name="id", type=Type.string),
            "source": Field(name="source", type=Type.string),
        }


class Manifest:
    """
    Class representing a Fondant manifest.

    Args:
        specification: The manifest specification as a Python dict
    """

    def __init__(self, specification: t.Dict[str, t.Any]) -> None:
        self._specification = copy.deepcopy(specification)
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate a manifest specification against the manifest schema.

        Raises: InvalidManifest when the manifest is not valid.
        """
        spec_data = pkgutil.get_data("fondant", "schemas/manifest.json")

        if spec_data is None:
            raise FileNotFoundError("schemas/manifest.json not found")
        else:
            spec_str = spec_data.decode("utf-8")
            spec_schema = json.loads(spec_str)

        base_uri = (Path(__file__).parent / "schemas").as_uri()
        resolver = RefResolver(base_uri=f"{base_uri}/", referrer=spec_schema)
        validator = Draft4Validator(spec_schema, resolver=resolver)

        try:
            validator.validate(self._specification)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidManifest.create_from(e)

    @classmethod
    def create(cls, *, base_path: str, run_id: str, component_id: str) -> "Manifest":
        """Create an empty manifest.

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
            "index": {"location": f"/index/{run_id}/{component_id}"},
            "subsets": {},
        }
        return cls(specification)

    @classmethod
    def from_file(cls, path: str) -> "Manifest":
        """Load the manifest from the file specified by the provided path."""
        with open(path, encoding="utf-8") as file_:
            specification = json.load(file_)
            return cls(specification)

    def to_file(self, path) -> None:
        """Dump the manifest to the file specified by the provided path."""
        with open(path, "w", encoding="utf-8") as file_:
            json.dump(self._specification, file_)

    def copy(self) -> "Manifest":
        """Return a deep copy of itself."""
        return self.__class__(copy.deepcopy(self._specification))

    @property
    def metadata(self) -> t.Dict[str, t.Any]:
        return self._specification["metadata"]

    def update_metadata(self, key: str, value: t.Any) -> None:
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
        """The subsets of the manifest as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: Subset(subset, base_path=self.base_path)
                for name, subset in self._specification["subsets"].items()
            }
        )

    def add_subset(
        self, name: str, fields: t.Iterable[t.Union[Field, t.Tuple[str, Type]]]
    ) -> None:
        if name in self._specification["subsets"]:
            raise ValueError(f"A subset with name {name} already exists")

        self._specification["subsets"][name] = {
            "location": f"/{name}/{self.run_id}/{self.component_id}",
            "fields": {name: {"type": type_.name} for name, type_ in fields},
        }

    def remove_subset(self, name: str) -> None:
        if name not in self._specification["subsets"]:
            raise ValueError(f"Subset {name} not found in specification")

        del self._specification["subsets"][name]

    def evolve(  # noqa : PLR0912 (too many branches)
        self, component_spec: ComponentSpec
    ) -> "Manifest":
        """Evolve the manifest based on the component spec. The resulting
        manifest is the expected result if the current manifest is provided
        to the component defined by the component spec.
        """
        evolved_manifest = self.copy()

        # Update `component_id` of the metadata
        component_id = component_spec.name.lower().replace(" ", "_")
        evolved_manifest.update_metadata(key="component_id", value=component_id)

        # Update index location as this is currently always rewritten
        evolved_manifest.index._specification[
            "location"
        ] = f"/index/{self.run_id}/{component_id}"

        # If additionalSubsets is False in consumes,
        # Remove all subsets from the manifest that are not listed
        if not component_spec.accepts_additional_subsets:
            for subset_name in evolved_manifest.subsets:
                if subset_name not in component_spec.consumes:
                    evolved_manifest.remove_subset(subset_name)

        # If additionalSubsets is False in produces,
        # Remove all subsets from the manifest that are not listed
        if not component_spec.outputs_additional_subsets:
            for subset_name in evolved_manifest.subsets:
                if subset_name not in component_spec.produces:
                    evolved_manifest.remove_subset(subset_name)

        # If additionalFields is False for a consumed subset,
        # Remove all fields from that subset that are not listed
        for subset_name, subset in component_spec.consumes.items():
            if subset_name in evolved_manifest.subsets:
                if not subset.additional_fields:
                    for field_name in evolved_manifest.subsets[subset_name].fields:
                        if field_name not in subset.fields:
                            evolved_manifest.subsets[subset_name].remove_field(
                                field_name
                            )

        # For each output subset defined in the component, add or update it
        for subset_name, subset in component_spec.produces.items():
            # Subset is already in manifest, update it
            if subset_name in evolved_manifest.subsets:
                # If additional fields are not allowed, remove the fields not defined in the
                # component spec produces section
                if not subset.additional_fields:
                    for field_name in evolved_manifest.subsets[subset_name].fields:
                        if field_name not in subset.fields:
                            evolved_manifest.subsets[subset_name].remove_field(
                                field_name
                            )

                # Add fields defined in the component spec produces section
                # Overwrite to persist changes to the field (eg. type of column)
                for field in subset.fields.values():
                    evolved_manifest.subsets[subset_name].add_field(
                        field.name, field.type, overwrite=True
                    )

                # Update subset location as this is currently always rewritten
                evolved_manifest.subsets[subset_name]._specification[
                    "location"
                ] = f"/{subset_name}/{self.run_id}/{component_id}"

            # Subset is not yet in manifest, add it
            else:
                evolved_manifest.add_subset(subset_name, subset.fields.values())

        return evolved_manifest

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"
