"""This module defines common schemas and datatypes used to define Fondant manifests, components
and pipelines.
"""

import typing as t

import pyarrow as pa

KubeflowCommandArguments = t.List[t.Union[str, t.Dict[str, str]]]

_TYPES: t.Dict[str, pa.DataType] = {
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
    "decimal": pa.decimal128(38),
    "time32": pa.time32("s"),
    "time64": pa.time64("us"),
    "timestamp": pa.timestamp("us"),
    "date32": pa.date32(),
    "date64": pa.date64(),
    "duration": pa.duration("us"),
    "string": pa.string(),
    "utf8": pa.utf8(),
    "binary": pa.binary(),
}


class Type:
    def __init__(self, data_type: t.Union[str, pa.DataType]):
        self.value = self._validate_data_type(data_type)

    @staticmethod
    def _validate_data_type(data_type: t.Union[str, pa.DataType]):
        """
        Validates the provided data type and returns the corresponding data type object.

        Args:
            data_type: The data type to validate.

        Returns:
            The validated `pa.DataType` object.
        """
        if isinstance(data_type, str):
            try:
                data_type = _TYPES[data_type]
            except KeyError:
                raise ValueError(
                    f"Invalid schema provided. Current available data types are:"
                    f" {_TYPES.keys()}"
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
            pa.list_(data_type.value if isinstance(data_type, Type) else data_type)
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
        if json_schema["type"] in _TYPES:
            return Type(json_schema["type"])

        elif json_schema["type"] == "array":
            items = json_schema["items"]
            if isinstance(items, dict):
                return cls.list(Type.from_json(items))
        else:
            raise ValueError(f"Invalid schema provided: {json_schema}")

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
