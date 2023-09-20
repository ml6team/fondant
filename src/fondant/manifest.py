"""This module defines classes to represent a Fondant manifest."""
import copy
import json
import pkgutil
import types
import typing as t
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonschema.exceptions
from fsspec import open as fs_open
from jsonschema import Draft4Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

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
                name: Field(name=name, type=Type.from_json(field))
                for name, field in self._specification["fields"].items()
            },
        )

    def add_field(self, name: str, type_: Type, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._specification["fields"]:
            msg = f"A field with name {name} already exists"
            raise ValueError(msg)

        self._specification["fields"][name] = type_.to_json()

    def remove_field(self, name: str) -> None:
        del self._specification["fields"][name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"


class Index(Subset):
    """Special case of a subset for the index, which has fixed fields."""

    @property
    def fields(self) -> t.Dict[str, Field]:
        return {
            "id": Field(name="id", type=Type("string")),
            "source": Field(name="source", type=Type("string")),
        }


@dataclass
class Metadata:
    """
    Class representing the Metadata of the manifest.

    Args:
        base_path: the base path used to store the artifacts
        pipeline_name: the name of the pipeline
        run_id: the run id of the pipeline
        component_id: the name of the component
        cache_key: the cache key of the component.
    """

    base_path: str
    pipeline_name: str
    run_id: str
    component_id: str
    cache_key: str

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)


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
            msg = "schemas/manifest.json not found"
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
            raise InvalidManifest.create_from(e)

    @classmethod
    def create(
        cls,
        *,
        pipeline_name: str,
        base_path: str,
        run_id: str,
        component_id: str,
        cache_key: str,
    ) -> "Manifest":
        """Create an empty manifest.

        Args:
            pipeline_name: the name of the pipeline
            base_path: The base path of the manifest
            run_id: The id of the current pipeline run
            component_id: The id of the current component being executed
            cache_key: The component cache key
        """
        metadata = Metadata(
            pipeline_name=pipeline_name,
            base_path=base_path,
            run_id=run_id,
            component_id=component_id,
            cache_key=cache_key,
        )

        specification = {
            "metadata": metadata.to_dict(),
            "index": {"location": f"/{pipeline_name}/{run_id}/{component_id}/index"},
            "subsets": {},
        }
        return cls(specification)

    @classmethod
    def from_file(cls, path: t.Union[str, Path]) -> "Manifest":
        """Load the manifest from the file specified by the provided path."""
        with fs_open(path, encoding="utf-8", auto_mkdir=True) as file_:
            specification = json.load(file_)
            return cls(specification)

    def to_file(self, path: t.Union[str, Path]) -> None:
        """Dump the manifest to the file specified by the provided path."""
        with fs_open(path, "w", encoding="utf-8", auto_mkdir=True) as file_:
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
    def pipeline_name(self) -> str:
        return self.metadata["pipeline_name"]

    @property
    def cache_key(self) -> str:
        return self.metadata["cache_key"]

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
            },
        )

    def add_subset(
        self,
        name: str,
        fields: t.Iterable[t.Union[Field, t.Tuple[str, Type]]],
    ) -> None:
        if name in self._specification["subsets"]:
            msg = f"A subset with name {name} already exists"
            raise ValueError(msg)

        self._specification["subsets"][name] = {
            "location": f"/{self.pipeline_name}/{self.run_id}/{self.component_id}/{name}",
            "fields": {name: type_.to_json() for name, type_ in fields},
        }

    def remove_subset(self, name: str) -> None:
        if name not in self._specification["subsets"]:
            msg = f"Subset {name} not found in specification"
            raise ValueError(msg)

        del self._specification["subsets"][name]

    def evolve(  # noqa : PLR0912 (too many branches)
        self,
        component_spec: ComponentSpec,
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
        ] = f"/{self.pipeline_name}/{self.run_id}/{component_id}/index"

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
            if subset_name in evolved_manifest.subsets and not subset.additional_fields:
                for field_name in evolved_manifest.subsets[subset_name].fields:
                    if field_name not in subset.fields:
                        evolved_manifest.subsets[subset_name].remove_field(
                            field_name,
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
                                field_name,
                            )

                # Add fields defined in the component spec produces section
                # Overwrite to persist changes to the field (eg. type of column)
                for field in subset.fields.values():
                    evolved_manifest.subsets[subset_name].add_field(
                        field.name,
                        field.type,
                        overwrite=True,
                    )

                # Update subset location as this is currently always rewritten
                evolved_manifest.subsets[subset_name]._specification[
                    "location"
                ] = f"/{self.pipeline_name}/{self.run_id}/{component_id}/{subset_name}"

            # Subset is not yet in manifest, add it
            else:
                evolved_manifest.add_subset(subset_name, subset.fields.values())

        return evolved_manifest

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"
