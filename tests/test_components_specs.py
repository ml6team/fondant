import json
import pytest
from express.exceptions import InvalidComponentSpec
from express.component_spec import ExpressComponent


def yaml_path() -> str:
    return "example_component.yaml"


VALID_COMPONENT_SPEC = {
    "input_subsets": {
        "subset1": {
            "fields": {
                "data": {"type": "bytes"},
                "height": {"type": "int"},
                "width": {"type": "int"}
            }
        },
        "subset2": {
            "fields": {
                "name": {"type": "str"},
                "age": {"type": "int"}
            }
        }
    },
    "output_subsets": {
        "subset1": {
            "fields": {
                "result": {"type": "str"}
            }
        },
        "subset2": {
            "fields": {
                "score": {"type": "int"},
                "grade": {"type": "str"}
            }
        }
    }
}

INVALID_COMPONENT_SPEC = {
    "input_subsets": {
        "subset1": {
            "fields": {
                "data": {"type": "bytes"},
                "height": {"type": None},
                "width": {"type": "int"}
            }
        },
        "subset2": {
            "fields": {
                "name": {"type": "string"},
                "age": {"type": "int"}
            }
        }
    },
    "output_subsets": {
        "subset1": {
            "fields": {
                "result": {"type": "str"}
            }
        },
        "subset2": {
            "fields": {
                "score": {"type": "int"},
                "grade": {"type": "str"}
            }
        }
    }
}

print(ExpressComponent(VALID_COMPONENT_SPEC, yaml_path()))


def test_component_spec_validation():
    """Test that the manifest is validated correctly on instantiation"""
    ExpressComponent(VALID_COMPONENT_SPEC, yaml_path())
    with pytest.raises(InvalidComponentSpec):
        ExpressComponent(INVALID_COMPONENT_SPEC, yaml_path())


def test_attribute_access():
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup
    """
    express_component = ExpressComponent(VALID_COMPONENT_SPEC, yaml_path())

    assert express_component.name == "Example component"
    assert express_component.description == "This is an example component"
    assert express_component.input_subsets['subset1'].fields["data"].type == "bytes"
