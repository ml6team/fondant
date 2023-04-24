"""Fondant component specs test"""
import os
import pytest
import yaml
from fondant.exceptions import InvalidComponentSpec
from fondant.component_spec import ComponentSpec

valid_path = os.path.join("tests/component_example", "valid_component")
invalid_path = os.path.join("tests/component_example", "invalid_component")


@pytest.fixture
def valid_fondant_schema() -> str:
    return os.path.join(valid_path, "fondant_component.yaml")


@pytest.fixture
def valid_kubeflow_schema() -> str:
    return os.path.join(valid_path, "kubeflow_component.yaml")


@pytest.fixture
def invalid_fondant_schema() -> str:
    return os.path.join(invalid_path, "fondant_component.yaml")


def test_component_spec_validation(valid_fondant_schema, invalid_fondant_schema):
    """Test that the manifest is validated correctly on instantiation"""
    ComponentSpec(valid_fondant_schema)
    with pytest.raises(InvalidComponentSpec):
        ComponentSpec(invalid_fondant_schema)


def test_attribute_access(valid_fondant_schema):
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup
    """
    fondant_component = ComponentSpec(valid_fondant_schema)

    assert fondant_component.name == "Example component"
    assert fondant_component.description == "This is an example component"
    assert fondant_component.input_subsets['images'].fields["data"].type == "binary"


def test_kfp_component_creation(valid_fondant_schema, valid_kubeflow_schema):
    """
    Test that the created kubeflow component matches the expected kubeflow component
    """
    fondant_component = ComponentSpec(valid_fondant_schema)
    kubeflow_schema = yaml.safe_load(open(valid_kubeflow_schema, 'r'))
    assert fondant_component.kubeflow_component_specification == kubeflow_schema
