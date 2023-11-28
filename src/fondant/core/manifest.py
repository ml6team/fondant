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

from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidManifest
from fondant.core.schema import Field, Type


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
        spec_data = pkgutil.get_data("fondant", "core/schemas/manifest.json")

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

        registry = Registry(retrieve=retrieve_from_filesystem)  # type: ignore
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
            "index": {"location": f"/{component_id}"},
            "fields": {},
        }
        return cls(specification)

    @classmethod
    def from_file(cls, path: t.Union[str, Path]) -> "Manifest":
        """Load the manifest from the file specified by the provided path."""
        with fs_open(path, encoding="utf-8") as file_:
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

    @property
    def index(self) -> Field:
        return Field(name="id", location=self._specification["index"]["location"])

    def update_metadata(self, key: str, value: t.Any) -> None:
        self.metadata[key] = value

    @property
    def base_path(self) -> str:
        return self.metadata["base_path"]

    def get_field_location(self, field_name: str):
        """Return absolute path to the field location."""
        if field_name == "id":
            return f"{self.base_path}/{self.pipeline_name}/{self.run_id}{self.index.location}"
        if field_name not in self.fields:
            msg = f"Field {field_name} is not available in the manifest."
            raise ValueError(msg)

        field = self.fields[field_name]
        return f"{self.base_path}/{self.pipeline_name}/{self.run_id}{field.location}"

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
    def fields(self) -> t.Mapping[str, Field]:
        """The fields of the manifest as an immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(
                    name=name,
                    type=Type.from_json(field),
                    location=field["location"],
                )
                for name, field in self._specification["fields"].items()
            },
        )

    def add_or_update_field(self, field: Field, overwrite: bool = False):
        """Add or update field to manifest."""
        if field.name == "index":
            self._add_or_update_index(field, overwrite=True)
        elif overwrite is False and field.name in self._specification["fields"]:
            msg = (
                f"A field with name {field.name} already exists. Set overwrite to true, "
                f"if you want to update the field."
            )
            raise ValueError(msg)
        else:
            self._specification["fields"][field.name] = {
                "location": f"/{self.component_id}",
                **field.type.to_json(),
            }

    def _add_or_update_index(self, field: Field, overwrite: bool = True):
        """Add or update the manifest index."""
        if overwrite is False:
            msg = (
                "The index already exists. Set overwrite to true, "
                "if you want to update the index."
            )
            raise ValueError(msg)

        if field.name != "index":
            msg = (
                f"The field name is {field.name}. If you try to update the index, "  # nosec B608
                f"set the field name to `index`."
            )
            raise ValueError(msg)

        self._specification["index"] = {
            "location": f"/{field.location}",
        }

    def remove_field(self, name: str) -> None:
        if name not in self._specification["fields"]:
            msg = f"Field {name} not found in specification"
            raise ValueError(msg)

        del self._specification["fields"][name]

    def evolve(  # : PLR0912 (too many branches)
        self,
        component_spec: ComponentSpec,
        *,
        run_id: t.Optional[str] = None,
    ) -> "Manifest":
        """Evolve the manifest based on the component spec. The resulting
        manifest is the expected result if the current manifest is provided
        to the component defined by the component spec.

        Args:
            component_spec: the component spec
            run_id: the run id to include in the evolved manifest. If no run id is provided,
            the run id from the original manifest is propagated.
        """
        evolved_manifest = self.copy()

        # Update `component_id` of the metadata
        component_id = component_spec.component_folder_name
        evolved_manifest.update_metadata(key="component_id", value=component_id)

        if run_id is not None:
            evolved_manifest.update_metadata(key="run_id", value=run_id)

        # Update index location as this is always rewritten
        evolved_manifest.add_or_update_field(
            Field(name="index", location=component_spec.component_folder_name),
        )

        # Remove all previous fields if the component changes the index
        if component_spec.previous_index:
            for field_name in evolved_manifest.fields:
                evolved_manifest.remove_field(field_name)

        # Add or update all produced fields defined in the component spec
        for name, field in component_spec.produces.items():
            # If field was not part of the input manifest, add field to output manifest.
            # If field was part of the input manifest and got produced by the component, update
            # the manifest field.
            evolved_manifest.add_or_update_field(field, overwrite=True)

        return evolved_manifest

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"
