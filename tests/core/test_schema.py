import pyarrow as pa
import pytest
from fondant.core.exceptions import InvalidTypeSchema
from fondant.core.schema import Type


def test_valid_type():
    """Test that a data type specified with the Type class matches the expected pyarrow schema."""
    assert Type("int8").name == "int8"
    assert Type("int8").value == pa.int8()
    assert Type.list(Type("int8")).value == pa.list_(pa.int8())
    assert Type.list(Type.list(Type("string"))).value == pa.list_(pa.list_(pa.string()))
    assert Type("int8").to_dict() == {"type": "int8"}
    assert Type.list("float32").to_dict() == {
        "type": "array",
        "items": {"type": "float32"},
    }
    assert Type.struct(
        [
            ("f1", "int32"),
            ("f2", "string"),
            ("f3", Type.list("int8")),
            ("f4", Type.struct([("f5", "int32")])),
        ],
    ).to_dict() == {
        "type": "object",
        "properties": {
            "f1": {"type": "int32"},
            "f2": {"type": "string"},
            "f3": {"type": "array", "items": {"type": "int8"}},
            "f4": {"type": "object", "properties": {"f5": {"type": "int32"}}},
        },
    }


def test_valid_json_schema():
    """Test that Type class initialized with a json schema matches the expected pyarrow schema."""
    assert Type.from_dict({"type": "string"}).value == pa.string()
    assert Type.from_dict(
        {"type": "array", "items": {"type": "int8"}},
    ).value == pa.list_(pa.int8())
    assert Type.from_dict(
        {"type": "array", "items": {"type": "array", "items": {"type": "int8"}}},
    ).value == pa.list_(pa.list_(pa.int8()))
    assert Type.from_dict(
        {
            "type": "object",
            "properties": {
                "f1": {"type": "int32"},
                "f2": {"type": "string"},
                "f3": {"type": "array", "items": {"type": "int8"}},
                "f4": {"type": "object", "properties": {"f5": {"type": "int32"}}},
            },
        },
    ) == Type.struct(
        [
            ("f1", "int32"),
            ("f2", "string"),
            ("f3", Type.list("int8")),
            ("f4", Type.struct([("f5", "int32")])),
        ],
    )


@pytest.mark.parametrize(
    "statement",
    [
        'Type("invalid_type")',
        'Type("invalid_type").to_dict()',
        'Type.list(Type("invalid_type"))',
        'Type.list(Type("invalid_type")).to_dict()',
        'Type.from_dict({"type": "invalid_value"})',
        'Type.from_dict({"type": "invalid_value", "items": {"type": "int8"}})',
        'Type.from_dict({"type": "array", "items": {"type": "invalid_type"}})',
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
