import pyarrow as pa
import pytest
from fondant.core.exceptions import InvalidTypeSchema
from fondant.core.schema import Type, dict_to_produces, produces_to_dict


def test_produces_parsing():
    """Test that the produces argument can be properly serialized and deserialized."""
    produces = {
        "text": pa.string(),
        "embedding": pa.list_(pa.int32()),
    }
    expected_produces_to_dict = {
        "text": {"type": "string"},
        "embedding": {"type": "array", "items": {"type": "int32"}},
    }
    actual_produces_to_dict = produces_to_dict(produces)
    assert actual_produces_to_dict == expected_produces_to_dict
    assert dict_to_produces(actual_produces_to_dict) == produces


def test_valid_type():
    """Test that a data type specified with the Type class matches the expected pyarrow schema."""
    assert Type("int8").name == "int8"
    assert Type("int8").value == pa.int8()
    assert Type.list(Type("int8")).value == pa.list_(pa.int8())
    assert Type.list(Type.list(Type("string"))).value == pa.list_(pa.list_(pa.string()))
    assert Type("int8").to_json() == {"type": "int8"}
    assert Type.list("float32").to_json() == {
        "type": "array",
        "items": {"type": "float32"},
    }


def test_valid_json_schema():
    """Test that Type class initialized with a json schema matches the expected pyarrow schema."""
    assert Type.from_json({"type": "string"}).value == pa.string()
    assert Type.from_json(
        {"type": "array", "items": {"type": "int8"}},
    ).value == pa.list_(pa.int8())
    assert Type.from_json(
        {"type": "array", "items": {"type": "array", "items": {"type": "int8"}}},
    ).value == pa.list_(pa.list_(pa.int8()))


@pytest.mark.parametrize(
    "statement",
    [
        'Type("invalid_type")',
        'Type("invalid_type").to_json()',
        'Type.list(Type("invalid_type"))',
        'Type.list(Type("invalid_type")).to_json()',
        'Type.from_json({"type": "invalid_value"})',
        'Type.from_json({"type": "invalid_value", "items": {"type": "int8"}})',
        'Type.from_json({"type": "array", "items": {"type": "invalid_type"}})',
    ],
)
def test_invalid_json_schema(statement):
    """Test that an invalid type or schema specified with the Type class raise an invalid type
    schema error.
    """
    with pytest.raises(InvalidTypeSchema):
        eval(statement)


def test_equality():
    """Test equality function to compare schemas."""
    assert Type("int8") == Type("int8")
    assert Type("int8") != Type("float64")
    assert Type("int8").__eq__(Type("int8")) is True
    assert Type("int8").__eq__(Type("float64")) is False


def test_inequality_wrong_type():
    """Test inequality between different types."""
    assert Type("int8").__eq__("int8") is False
    assert Type("float64").__eq__("float64") is False
    assert Type("int8").__eq__(5) is False
    assert Type("float64").__eq__(0.9) is False
