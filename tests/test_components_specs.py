"""Express component specs test"""
import os
import pytest
from express.exceptions import InvalidComponentSpec
from express.component_spec import ExpressComponent

valid_path = os.path.join("component_example", "valid_component")
invalid_path = os.path.join("component_example", "invalid_component")


def valid_express_schema() -> str:
    return os.path.join(valid_path, "express_component.yaml")


def valid_kubeflow_schema() -> str:
    return os.path.join(valid_path, "kubeflow_component.yaml")


def invalid_express_schema() -> str:
    return os.path.join(invalid_path, "express_component.yaml")


def test_component_spec_validation():
    """Test that the manifest is validated correctly on instantiation"""
    ExpressComponent(valid_express_schema())
    with pytest.raises(InvalidComponentSpec):
        ExpressComponent(invalid_express_schema())


def test_attribute_access():
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup
    """
    express_component = ExpressComponent(valid_express_schema())

    assert express_component.name == "Example component"
    assert express_component.description == "This is an example component"
    assert express_component.input_subsets['images'].fields["data"].type == "binary"
