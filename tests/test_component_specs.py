"""Fondant component specs test."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fondant.component_spec import (
    ColumnMapping,
    ComponentSpec,
    ComponentSubset,
    KubeflowComponentSpec,
    SpecMapper,
)
from fondant.exceptions import InvalidComponentSpec, InvalidSubsetMapping
from fondant.schema import Type

component_specs_path = Path(__file__).parent / "example_specs/component_specs"


@pytest.fixture()
def valid_fondant_schema() -> dict:
    with open(component_specs_path / "valid_component.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def valid_fondant_schema_no_args() -> dict:
    with open(component_specs_path / "valid_component_no_args.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def valid_kubeflow_schema() -> dict:
    with open(component_specs_path / "kubeflow_component.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def invalid_fondant_schema() -> dict:
    with open(component_specs_path / "invalid_component.yaml") as f:
        return yaml.safe_load(f)


@patch("pkgutil.get_data", return_value=None)
def test_component_spec_pkgutil_error(mock_get_data):
    """Test that FileNotFoundError is raised when pkgutil.get_data returns None."""
    with pytest.raises(FileNotFoundError):
        ComponentSpec("example_component.yaml")


def test_component_spec_validation(valid_fondant_schema, invalid_fondant_schema):
    """Test that the manifest is validated correctly on instantiation."""
    ComponentSpec(valid_fondant_schema)
    with pytest.raises(InvalidComponentSpec):
        ComponentSpec(invalid_fondant_schema)


def test_attribute_access(valid_fondant_schema):
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup.
    """
    fondant_component = ComponentSpec(valid_fondant_schema)

    assert fondant_component.name == "Example component"
    assert fondant_component.description == "This is an example component"
    assert fondant_component.consumes["images"].fields["data"].type == Type("binary")
    assert fondant_component.consumes["embeddings"].fields["data"].type == Type.list(
        Type("float32"),
    )


def test_kfp_component_creation(valid_fondant_schema, valid_kubeflow_schema):
    """Test that the created kubeflow component matches the expected kubeflow component."""
    fondant_component = ComponentSpec(valid_fondant_schema)
    kubeflow_component = fondant_component.kubeflow_specification
    assert kubeflow_component._specification == valid_kubeflow_schema


def test_subset_field_mapping():
    """Test that the subset field mapper returns a valid mapping dictionary."""
    column_mapping_list = [
        ColumnMapping(dataset_column="images_data", component_column="picture_array"),
        ColumnMapping(dataset_column="images_size", component_column="picture_area"),
    ]
    column_mapping = ColumnMapping.list_to_dict(column_mapping_list)
    mapper = SpecMapper.from_dict(column_mapping)
    expected_mapping = {
        ("images", "data"): ("picture", "array"),
        ("images", "size"): ("picture", "area"),
    }

    assert mapper.subset_field_mapping == expected_mapping


def test_invalid_subset_field_mapping():
    """Test that the subset field mapper returns a valid error when the mapping dictionary
    contains conflicting mapping between subsets.
    """
    """Test that the subset field mapper returns a valid mapping dictionary."""
    invalid_examples = [
        [
            ColumnMapping(
                dataset_column="images_data",
                component_column="picture_array",
            ),
            ColumnMapping(dataset_column="images_size", component_column="images_area"),
        ],
        [
            ColumnMapping(
                dataset_column="images_data",
                component_column="picture_array",
            ),
            ColumnMapping(dataset_column="photo_size", component_column="picture_area"),
        ],
    ]

    for invalid_example in invalid_examples:
        column_mapping = ColumnMapping.list_to_dict(invalid_example)
        with pytest.raises(InvalidSubsetMapping):
            SpecMapper.from_dict(column_mapping)


def test_component_spec_no_args(valid_fondant_schema_no_args):
    """Test that a component spec without args is supported."""
    fondant_component = ComponentSpec(valid_fondant_schema_no_args)

    assert fondant_component.name == "Example component"
    assert fondant_component.description == "This is an example component"
    assert fondant_component.args == {}


def test_component_spec_to_file(valid_fondant_schema):
    """Test that the ComponentSpec can be written to a file."""
    component_spec = ComponentSpec(valid_fondant_schema)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "component_spec.yaml")
        component_spec.to_file(file_path)

        with open(file_path) as f:
            written_data = yaml.safe_load(f)

        # check if the written data is the same as the original data
        assert written_data == valid_fondant_schema


def test_component_column_mapping(valid_fondant_schema):
    """Test the component spec subset mapping method."""
    column_mapping_list = [
        ColumnMapping(dataset_column="pictures_array", component_column="images_data"),
    ]
    column_mapping = ColumnMapping.list_to_dict(column_mapping_list)

    original_component_spec = ComponentSpec(valid_fondant_schema)
    mapped_component_spec = ComponentSpec(
        valid_fondant_schema,
        column_mapping=column_mapping,
    )

    assert (
        original_component_spec.consumes["images"].fields["data"].type
        == mapped_component_spec.consumes["pictures"].fields["array"].type
    )

    # Non existing subsets/fields
    invalid_column_mappings = [
        [
            ColumnMapping(
                dataset_column="non+existing+subset_data",
                component_column="pictures_array",
            ),
        ],
        [
            ColumnMapping(
                dataset_column="images_non+existing+field`",
                component_column="pictures_array",
            ),
        ],
    ]

    for invalid_column_mapping in invalid_column_mappings:
        column_mapping = ColumnMapping.list_to_dict(invalid_column_mapping)
        with pytest.raises(InvalidComponentSpec):
            ComponentSpec(
                valid_fondant_schema,
                column_mapping=column_mapping,
            )

    # Conflicting subset mapping
    invalid_column_mappings = [
        [
            ColumnMapping(
                dataset_column="images_data",
                component_column="embeddings_data",
            ),
        ],
    ]
    for invalid_column_mapping in invalid_column_mappings:
        column_mapping = ColumnMapping.list_to_dict(invalid_column_mapping)
    with pytest.raises(InvalidSubsetMapping):
        ComponentSpec(
            valid_fondant_schema,
            column_mapping=column_mapping,
        )


def test_kubeflow_component_spec_to_file(valid_kubeflow_schema):
    """Test that the KubeflowComponentSpec can be written to a file."""
    kubeflow_component_spec = KubeflowComponentSpec(valid_kubeflow_schema)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "kubeflow_component_spec.yaml")
        kubeflow_component_spec.to_file(file_path)

        with open(file_path) as f:
            written_data = yaml.safe_load(f)

        # check if the written data is the same as the original data
        assert written_data == valid_kubeflow_schema


def test_component_spec_repr(valid_fondant_schema):
    """Test that the __repr__ method of ComponentSpec returns the expected string."""
    fondant_component = ComponentSpec(valid_fondant_schema)
    expected_repr = f"ComponentSpec({valid_fondant_schema!r})"
    assert repr(fondant_component) == expected_repr


def test_kubeflow_component_spec_repr(valid_kubeflow_schema):
    """Test that the __repr__ method of KubeflowComponentSpec returns the expected string."""
    kubeflow_component_spec = KubeflowComponentSpec(valid_kubeflow_schema)
    expected_repr = f"KubeflowComponentSpec({valid_kubeflow_schema!r})"
    assert repr(kubeflow_component_spec) == expected_repr


def test_component_subset_repr():
    """Test that the __repr__ method of ComponentSubset returns the expected string."""
    component_subset_schema = {
        "name": "Example subset",
        "description": "This is an example subset",
    }

    component_subset = ComponentSubset(component_subset_schema)
    expected_repr = f"ComponentSubset({component_subset_schema!r})"
    assert repr(component_subset) == expected_repr
