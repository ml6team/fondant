"""This module defines common schemas and datatypes used to define Fondant manifests, components
and pipelines.
"""

import re
import typing as t

import pyarrow as pa

from fondant.exceptions import InvalidTypeSchema

KubeflowCommandArguments = t.List[t.Union[str, t.Dict[str, str]]]

"""
Types based on:
- https://arrow.apache.org/docs/python/api/datatypes.html#api-types
"""
_TYPES: t.Dict[str, pa.DataType] = {
    "null": pa.null(),
    "bool": pa.bool_(),
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "decimal128": pa.decimal128(38),
    "time32": pa.time32("s"),
    "time64": pa.time64("us"),
    "timestamp": pa.timestamp("us"),
    "date32": pa.date32(),
    "date64": pa.date64(),
    "duration": pa.duration("us"),
    "string": pa.string(),
    "utf8": pa.utf8(),
    "binary": pa.binary(),
    "large_binary": pa.large_binary(),
    "large_utf8": pa.large_utf8(),
}


class Type:
    """
    The `Type` class provides a way to define and validate data types for various purposes. It
      supports different data types including primitive types and complex types like lists.
    """

    def __init__(self, data_type: t.Union[str, pa.DataType]):
        self.value = self._validate_data_type(data_type)

    @staticmethod
    def _validate_data_type(data_type: t.Union[str, pa.DataType]) -> pa.DataType:
        """
        Validates the provided data type and returns the corresponding data type object.

        Args:
            data_type: The data type to validate.

        Returns:
            The validated `pa.DataType` object.
        """
        if not isinstance(data_type, (Type, pa.DataType)):
            try:
                data_type = _TYPES[data_type]
            except KeyError:
                msg = (
                    f"Invalid schema provided {data_type} with type {type(data_type)}. "
                    f"Current available data types are: {_TYPES.keys()}"
                )
                raise InvalidTypeSchema(
                    msg,
                )
        return data_type

    @classmethod
    def list(cls, data_type: t.Union[str, pa.DataType, "Type"]) -> "Type":
        """
        Creates a new `Type` instance representing a list of the specified data type.

        Args:
            data_type: The data type for the list elements. It can be a string representing the
            data type or an existing `pa.DataType` object.

        Returns:
            A new `Type` instance representing a list of the specified data type.
        """
        data_type = cls._validate_data_type(data_type)
        return cls(
            pa.list_(data_type.value if isinstance(data_type, Type) else data_type),
        )

    @classmethod
    def from_json(cls, json_schema: dict):
        """
        Creates a new `Type` instance based on a dictionary representation of the json schema
          of a data type (https://swagger.io/docs/specification/data-models/data-types/).

        Args:
            json_schema: The dictionary representation of the data type, can represent nested values

        Returns:
            A new `Type` instance representing the specified data type.
        """
        if json_schema["type"] == "array":
            items = json_schema["items"]
            if isinstance(items, dict):
                return cls.list(cls.from_json(items))
            return None

        return cls(json_schema["type"])

    def to_json(self) -> dict:
        """
        Converts the `Type` instance to its JSON representation.

        Returns:
            A dictionary representing the JSON schema of the data type.
        """
        if isinstance(self.value, pa.ListType):
            items = self.value.value_type
            if isinstance(items, pa.DataType):
                return {"type": "array", "items": Type(items).to_json()}

        type_ = None
        for type_name, data_type in _TYPES.items():
            if self.value.equals(data_type):
                type_ = type_name
                break

        return {"type": type_}

    @property
    def name(self):
        """Name of the data type."""
        return str(self.value)

    def __repr__(self):
        """Returns a string representation of the `Type` instance."""
        return f"Type({repr(self.value)})"

    def __eq__(self, other):
        if isinstance(other, Type):
            return self.value == other.value

        return False


class Field(t.NamedTuple):
    """Class representing a single field or column in a Fondant subset."""

    name: str
    type: Type


def validate_partition_number(arg_value):
    if arg_value in ["disable", None, "None"]:
        return arg_value if arg_value != "None" else None
    try:
        return int(arg_value)
    except ValueError:
        msg = f"Invalid format for '{arg_value}'. The value must be an integer or set to 'disable'"
        raise InvalidTypeSchema(msg)


def validate_partition_size(arg_value):
    if arg_value in ["disable", None, "None"]:
        return arg_value if arg_value != "None" else None

    file_size_pattern = r"^\d+(?:\.\d+)?(?:KB|MB|GB|TB)$"
    if not bool(re.match(file_size_pattern, arg_value, re.I)):
        msg = (
            f"Invalid partition size defined `{arg_value}`, partition size must be a string f"
            f"ollowed by a file size notation e.g. ('250MB') or set to 'disable' to disable"
            f" the automatic partitioning"
        )
        raise InvalidTypeSchema(msg)
    return arg_value
