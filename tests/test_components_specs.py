import json
import pytest
from express.exceptions import InvalidComponentSpec
from express.component_spec import ExpressComponent


def yaml_path() -> str:
    return "example_component.yaml"


VALID_COMPONENT_SPEC = {
    "input_subsets": {
        "images": {
            "fields": {
                "data": {"type": "binary"},
                "height": {"type": "int32"},
                "width": {"type": "int32"}
            }
        },
        "texts": {
            "fields": {
                "data": {"type": "utf8"},
                "length": {"type": "int32"}
            }
        }
    },
    "output_subsets": {
        "embedding": {
            "fields": {
                "data": {"type": "binary"}
            }
        },
        "captions": {
            "fields": {
                "length": {"type": "int32"},
                "language": {"type": "utf8"}
            }
        }
    }
}

INVALID_COMPONENT_SPEC = {
    "input_subsets": {
        "images": {
            "fields": {
                "data": {"type": "binary"},
                "height": {"type": None},
                "width": {"type": "int32"}
            }
        },
        "texts": {
            "fields": {
                "data": {"type": "utf8"},
                "length": {"type": "int32"}
            }
        }
    },
    "output_subsets": {
        "embedding": {
            "fields": {
                "data": {"type": "int32"}
            }
        },
        "captions": {
            "fields": {
                "length": {"type": "int32"},
                "language": {"type": "str"}
            }
        }
    }
}


def test_component_spec_validation():
    """Test that the manifest is validated correctly on instantiation"""
    ExpressComponent(yaml_path(), VALID_COMPONENT_SPEC)
    with pytest.raises(InvalidComponentSpec):
        ExpressComponent(yaml_path(), INVALID_COMPONENT_SPEC)


def test_attribute_access():
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup
    """
    express_component = ExpressComponent(yaml_path(), VALID_COMPONENT_SPEC)

    assert express_component.name == "Example component"
    assert express_component.description == "This is an example component"
    assert express_component.input_subsets['images'].fields["data"].type == "binary"
