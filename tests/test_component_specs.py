"""Fondant component specs test"""
import pytest
import yaml
from pathlib import Path

from fondant.exceptions import InvalidComponentSpec
from fondant.component_spec import FondantComponentSpec
from fondant.schema import Type

component_specs_path = Path(__file__).parent / "example_specs/component_specs"


@pytest.fixture
def valid_fondant_schema() -> dict:
    with open(component_specs_path / "valid_component.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def valid_kubeflow_schema() -> dict:
    with open(component_specs_path / "kubeflow_component.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def invalid_fondant_schema() -> dict:
    with open(component_specs_path / "invalid_component.yaml") as f:
        return yaml.safe_load(f)


def test_component_spec_validation(valid_fondant_schema, invalid_fondant_schema):
    """Test that the manifest is validated correctly on instantiation"""
    FondantComponentSpec(valid_fondant_schema)
    with pytest.raises(InvalidComponentSpec):
        FondantComponentSpec(invalid_fondant_schema)


def test_attribute_access(valid_fondant_schema):
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup
    """
    fondant_component = FondantComponentSpec(valid_fondant_schema)

    assert fondant_component.name == "Example component"
    assert fondant_component.description == "This is an example component"
    assert fondant_component.input_subsets["images"].fields["data"].type == Type.binary


def test_kfp_component_creation(valid_fondant_schema, valid_kubeflow_schema):
    """
    Test that the created kubeflow component matches the expected kubeflow component
    """
    fondant_component = FondantComponentSpec(valid_fondant_schema)
    kubeflow_component = fondant_component.kubeflow_specification
    assert kubeflow_component._specification == valid_kubeflow_schema
