import pyarrow as pa
import pytest

from fondant.schema import Type


def test_valid_type():
    """Test that a data type specified with the Type class matches the expected pyarrow schema."""
    assert Type("int8").value == pa.int8()
    assert Type.list(Type("int8")).value == pa.list_(pa.int8())
    assert Type.list(Type.list(Type("string"))).value == pa.list_(pa.list_(pa.string()))
    assert Type("int8").to_json() == {"type": "int8"}
    assert Type.list("float32").to_json() == {
        "type": "array",
        "items": {"type": "float32"},
    }


def test_valid_json_schema():
    """Test that a json schema specified with the Type class matches the expected pyarrow schema."""
    assert Type.from_json({"type": "string"}).value == pa.string()
    assert Type.from_json(
        {"type": "array", "items": {"type": "int8"}}
    ).value == pa.list_(pa.int8())
    assert Type.from_json(
        {"type": "array", "items": {"type": "array", "items": {"type": "int8"}}}
    ).value == pa.list_(pa.list_(pa.int8()))


def test_invalid_json_schema():
    """Test that an invalid type specified with the Type class raise an error."""
    with pytest.raises(ValueError):
        Type("invalid_type")
        Type.list(Type("invalid_type"))
        Type.from_json({"type": "invalid_value", "items": {"type": "int8"}})
        Type.from_json({"type": "array", "items": {"type": "invalid_type"}})
