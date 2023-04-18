"""Express component specs test"""
import os
import pytest
import yaml
from express.exceptions import InvalidComponentSpec
from express.component_spec import ExpressComponent

valid_path = os.path.join("component_example", "valid_component")
invalid_path = os.path.join("component_example", "invalid_component")


@pytest.fixture
def valid_express_schema() -> str:
    return os.path.join(valid_path, "express_component.yaml")


@pytest.fixture
def valid_kubeflow_schema() -> str:
    return os.path.join(valid_path, "kubeflow_component.yaml")


@pytest.fixture
def invalid_express_schema() -> str:
    return os.path.join(invalid_path, "express_component.yaml")


def test_component_spec_validation(valid_express_schema, invalid_express_schema):
    """Test that the manifest is validated correctly on instantiation"""
    ExpressComponent(valid_express_schema)
    with pytest.raises(InvalidComponentSpec):
        ExpressComponent(invalid_express_schema)


def test_attribute_access(valid_express_schema):
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup
    """
    express_component = ExpressComponent(valid_express_schema)

    assert express_component.name == "Example component"
    assert express_component.description == "This is an example component"
    assert express_component.input_subsets['images'].fields["data"].type == "binary"


def test_kfp_component_creation(valid_express_schema, valid_kubeflow_schema):
    """
    Test that the created kubeflow component matches the expected kubeflow component
    """
    express_component = ExpressComponent(valid_express_schema)
    kubeflow_schema = yaml.safe_load(open(valid_kubeflow_schema, 'r'))
    assert express_component.kubeflow_component_specification == kubeflow_schema
