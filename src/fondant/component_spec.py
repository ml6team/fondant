"""This module defines classes to represent an Fondant component specification."""
import ast
import copy
import json
import pkgutil
import types
import typing as t
from dataclasses import dataclass
from pathlib import Path

import jsonschema.exceptions
import yaml
from jsonschema import Draft4Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

from fondant.exceptions import InvalidComponentSpec, InvalidSubsetMapping
from fondant.schema import Field, KubeflowCommandArguments, Type

# TODO: remove after upgrading to kfpv2
kubeflow_to_python_type_dict = {
    "String": str,
    "Integer": int,
    "Float": float,
    "Boolean": ast.literal_eval,
    "JsonObject": json.loads,
    "JsonArray": json.loads,
}


def kubeflow2python_type(type_: str) -> t.Any:
    map_fn = kubeflow_to_python_type_dict[type_]
    return lambda value: map_fn(value) if value != "None" else None  # type: ignore


# TODO: Change after upgrading to kfp v2
# :https://www.kubeflow.org/docs/components/pipelines/v2/data-types/parameters/
python2kubeflow_type = {
    "str": "String",
    "int": "Integer",
    "float": "Float",
    "bool": "Boolean",
    "dict": "JsonObject",
    "list": "JsonArray",
}


@dataclass
class Argument:
    """
    Kubeflow component argument.

    Args:
        name: name of the argument
        description: argument description
        type: the python argument type (str, int, ...)
        default: default value of the argument (defaults to None)
    """

    name: str
    description: str
    type: str
    default: t.Optional[str] = None


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


@dataclass
class ColumnMapping:
    """
    Represents a column mapping between a source dataset column and a target component column.
    This class is used to facilitate the alignment of column names between different datasets
    and component specifications.

    Args:
        dataset_column: The name of the source column in your dataset that you want to map with
          format <subset_field>.
        component_column: The name of the corresponding target column in a component's
          specification that you want to align with.

    Example:
        Consider a scenario where your dataset has a column named 'images_data', but a specific
        component you are using requires the column to be named 'images_array'. You can create a
        ColumnMapping instance to link 'images_data' to 'images_array', ensuring alignment
        between your dataset and the component's requirements.
    """

    dataset_column: str
    component_column: str

    @classmethod
    def list_to_dict(
        cls,
        column_mapping_list: t.List["ColumnMapping"],
    ) -> t.Dict[str, str]:
        """
        Convert a list of ColumnMapping instances to a dictionary where keys are dataset column
        names and values are corresponding component column names.

        Args:
            column_mapping_list (list): List of ColumnMapping instances.

        Returns:
            dict: A dictionary containing the mapping of dataset columns to component columns.
        """
        column_mapping = {}
        for mapping in column_mapping_list:
            column_mapping[mapping.dataset_column] = mapping.component_column
        return column_mapping


class SpecMapper:
    """Class that represents a mapping between different subsets and fields."""

    def __init__(self):
        self.subset_field_mapping = {}

    def __setitem__(
        self,
        source_subset_field: t.Tuple[str, str],
        target_subset_field: t.Tuple[str, str],
    ) -> None:
        """
        Add a mapping from a source subset and field to a target subset and field.

        Args:
            source_subset_field: The source subset and field as a tuple.
            target_subset_field: The target subset and field as a tuple.
        """
        source_subset, source_field = source_subset_field
        target_subset, target_field = target_subset_field

        self._check_for_conflicting_mapping(source_subset, target_subset)

        self.subset_field_mapping[source_subset_field] = target_subset_field

    def __getitem__(
        self,
        source_subset_field: t.Tuple[str, str],
    ) -> t.Union[t.Tuple[str, str], None]:
        """
        Retrieve the mapping for the given source subset and field.

        Args:
            source_subset_field: The source subset and field as a tuple.

        Returns:
            The corresponding target subset and field.
        """
        return self.subset_field_mapping.get(source_subset_field)

    def _check_for_conflicting_mapping(
        self,
        source_subset: str,
        target_subset: str,
    ) -> None:
        """
        Check if the added column is used in any existing mapping with a different source column.

        Args:
            source_subset: The name of the source subset to add.
            target_subset: The name of the target subset to add.
        """
        for mapped_source, mapped_target in self.subset_field_mapping.items():
            mapped_source_column, mapped_source_field = mapped_source
            mapped_target_column, mapped_target_field = mapped_target
            if (
                source_subset == mapped_source_column
                and target_subset != mapped_target_column
            ):
                msg = (
                    f"Conflicting mapping: Source column '{source_subset}' is already mapped to"
                    f" '{mapped_target_column}' cannot map it to '{target_subset}'."
                )
                raise InvalidSubsetMapping(msg)
            if (
                target_subset == mapped_target_column
                and source_subset != mapped_source_column
            ):
                msg = (
                    f"Conflicting mapping: target column '{target_subset}' "
                    f"is already mapped to '{mapped_source_column}` cannot map it to"
                    f" '{source_subset}'."
                )
                raise InvalidSubsetMapping(msg)

    @classmethod
    def from_dict(
        cls,
        column_mapping: t.Dict[str, str],
    ) -> "SpecMapper":
        """
        Create a SpecMapper instance from a dictionary containing remapping information.

        Args:
            column_mapping: A dictionary containing source subset-field mappings.

        Returns:
            A new instance of SpecMapper with the mappings from the remapping_dict.
        """
        subset_field_mapper = cls()
        for source_value, mapped_value in column_mapping.items():
            source_subset, source_field = source_value.rsplit("_")
            mapped_subset, mapped_field = mapped_value.rsplit("_")
            subset_field_mapper[(source_subset, source_field)] = (
                mapped_subset,
                mapped_field,
            )

        return subset_field_mapper


class ComponentSpec:
    """
    Class representing a Fondant component specification.

    Args:
        specification: The fondant component specification as a Python dict
        column_mapping: Optional dictionary that maps the column names of the consumed dataset to
         other column names that match a given component specification
    """

    def __init__(
        self,
        specification: t.Dict[str, t.Any],
        *,
        column_mapping: t.Optional[t.Dict[str, str]] = None,
    ) -> None:
        self._specification = copy.deepcopy(specification)
        if column_mapping:
            self._specification = self._remap_specification(column_mapping)
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

    def _remap_specification(
        self,
        column_mapping: t.Dict[str, str],
    ) -> t.Dict[str, t.Any]:
        """
        Remap fields and subsets of a component specification based on a remapping dictionary.

        Args:
            column_mapping: A dictionary that maps the column names of the consumed dataset to
         other column names that match a given component specification
        """

        def _validate_remapping_in_spec(mapper: SpecMapper):
            """Check that all the subsets and fields specified in the remapping dict exist in the
            component spec.
            """
            for source_subset, source_field in mapper.subset_field_mapping:
                if source_subset not in self._specification["consumes"]:
                    msg = (
                        f"`{source_subset}`does not exist in `{source_subset}` in the"
                        f" Component spec: \n {self._specification}"
                    )
                    raise InvalidComponentSpec(msg)

                if (
                    source_field
                    not in self._specification["consumes"][source_subset]["fields"]
                ):
                    msg = (
                        f"`{source_field}` field does not exist in `{source_subset}` "
                        f"subset of the component spec: \n {self._specification}"
                    )
                    raise InvalidComponentSpec(msg)

        def _add_defaults_to_mapper(mapper: SpecMapper):
            """
            Add default subset-field pairs to the mapper if no valid mapping pair is present in
            the component spec.
            """
            for subset_type in ["consumes", "produces"]:
                if subset_type in self._specification:
                    for subset_name, subset_fields in self._specification[
                        subset_type
                    ].items():
                        for field_name in subset_fields["fields"]:
                            if mapper[(subset_name, field_name)] is None:
                                mapper[(subset_name, field_name)] = (
                                    subset_name,
                                    field_name,
                                )

        def _get_mapped_specification(mapper: SpecMapper) -> t.Dict[str, t.Any]:
            """Map the specification based on the remapping dictionary."""
            modified_specification = copy.deepcopy(self._specification)

            for subset_type in ["consumes", "produces"]:
                if subset_type in self._specification:
                    modified_specification.pop(subset_type)

                    for subset_name, subset_fields in self._specification[
                        subset_type
                    ].items():
                        for field_name, field_schema in subset_fields["fields"].items():
                            mapped_subset_field = mapper[(subset_name, field_name)]

                            if mapped_subset_field is not None:
                                mapped_subset, mapped_field = mapped_subset_field
                                modified_specification.setdefault(
                                    subset_type,
                                    {},
                                ).setdefault(mapped_subset, {}).setdefault(
                                    "fields",
                                    {},
                                )[
                                    mapped_field
                                ] = field_schema

            return modified_specification

        # Create the mapper from the remapping dictionary
        column_mapping = {v: k for k, v in column_mapping.items()}
        spec_mapper = SpecMapper().from_dict(column_mapping)

        # Validate the remapping
        _validate_remapping_in_spec(mapper=spec_mapper)

        # Add default mappings to catch conflicting mapping across subsets
        _add_defaults_to_mapper(mapper=spec_mapper)

        # Get the remapped specification
        return _get_mapped_specification(mapper=spec_mapper)

    @classmethod
    def from_file(
        cls,
        path: t.Union[str, Path],
        column_mapping: t.Optional[t.Dict[str, str]] = None,
    ) -> "ComponentSpec":
        """Load the component spec from the file specified by the provided path."""
        with open(path, encoding="utf-8") as file_:
            specification = yaml.safe_load(file_)
            return cls(
                specification,
                column_mapping=column_mapping,
            )

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
        return {
            name: Argument(
                name=name,
                description=arg_info["description"],
                type=arg_info["type"],
                default=arg_info["default"] if "default" in arg_info else None,
            )
            for name, arg_info in self._specification.get("args", {}).items()
        }

    @property
    def specification(self) -> t.Dict[str, t.Any]:
        return copy.deepcopy(self._specification)

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

    @classmethod
    def from_fondant_component_spec(
        cls,
        fondant_component: ComponentSpec,
    ) -> "KubeflowComponentSpec":
        """Create a Kubeflow component spec from a Fondant component spec."""
        specification = {
            "name": fondant_component.name,
            "description": fondant_component.description,
            "inputs": [
                {
                    "name": "input_manifest_path",
                    "description": "Path to the input manifest",
                    "type": "String",
                },
                {
                    "name": "metadata",
                    "description": "Metadata arguments containing the run id and base path",
                    "type": "String",
                },
                {
                    "name": "component_spec",
                    "description": "The component specification as a dictionary",
                    "type": "JsonObject",
                    "default": "None",
                },
                {
                    "name": "input_partition_rows",
                    "description": "The number of rows to load per partition. Set to override the"
                    " automatic partitioning",
                    "type": "String",
                    "default": "None",
                },
                {
                    "name": "column_mapping",
                    "description": "A dictionary that maps the column names of the consumed"
                    " dataset to other column names that match a given"
                    " component specification",
                    "type": "JsonObject",
                    "default": "None",
                },
                *(
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "type": python2kubeflow_type[arg.type],
                        **({"default": arg.default} if arg.default is not None else {}),
                    }
                    for arg in fondant_component.args.values()
                ),
            ],
            "outputs": [
                {
                    "name": "output_manifest_path",
                    "description": "Path to the output manifest",
                    "type": "String",
                },
            ],
            "implementation": {
                "container": {
                    "image": fondant_component.image,
                    "command": [
                        "fondant",
                        "execute",
                        "main",
                        "--input_manifest_path",
                        {"inputPath": "input_manifest_path"},
                        "--metadata",
                        {"inputValue": "metadata"},
                        "--component_spec",
                        {"inputValue": "component_spec"},
                        "--input_partition_rows",
                        {"inputValue": "input_partition_rows"},
                        "--column_mapping",
                        {"inputValue": "column_mapping"},
                        *cls._dump_args(fondant_component.args.values()),
                        "--output_manifest_path",
                        {"outputPath": "output_manifest_path"},
                    ],
                },
            },
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
            dumped_args.append({"inputValue": arg_name})

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

    @property
    def input_arguments(self) -> t.Mapping[str, Argument]:
        """The input arguments of the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                info["name"]: Argument(
                    name=info["name"],
                    description=info["description"],
                    type=info["type"],
                    default=info["default"] if "default" in info else None,
                )
                for info in self._specification["inputs"]
            },
        )

    @property
    def output_arguments(self) -> t.Mapping[str, Argument]:
        """The output arguments of the component as an immutable mapping."""
        return types.MappingProxyType(
            {
                info["name"]: Argument(
                    name=info["name"],
                    description=info["description"],
                    type=info["type"],
                )
                for info in self._specification["outputs"]
            },
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r})"
