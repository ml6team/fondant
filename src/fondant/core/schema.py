"""This module defines common schemas and datatypes used to define Fondant manifests, components
and datasets.
"""

import os
import re
import typing as t
from dataclasses import dataclass
from enum import Enum, auto

import pyarrow as pa

from fondant.core.exceptions import InvalidTypeSchema


@dataclass
class DockerVolume:
    """Dataclass representing a DockerVolume.
    (https://docs.docker.com/compose/compose-file/05-services/#volumes).

    Args:
        type: the mount type volume (bind, volume)
        source: the source of the mount, a path on the host for a bind mount
        target: the path in the container where the volume is mounted.
    """

    type: str
    source: str
    target: str


class CloudCredentialsMount(Enum):
    AWS = auto()
    GCP = auto()
    AZURE = auto()

    def get_path(self):
        home_dir = os.path.expanduser("~")

        if self == CloudCredentialsMount.AWS:
            return f"{home_dir}/credentials:/root/.aws/credentials"

        if self == CloudCredentialsMount.GCP:
            return (
                f"{home_dir}/.config/gcloud/application_default_credentials.json:"
                f"/root/.config/gcloud/application_default_credentials.json"
            )

        if self == CloudCredentialsMount.AZURE:
            return f"{home_dir}/.azure:/root/.azure"

        return None


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
    "struct": pa.struct([]),
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
    def struct(
        cls,
        fields: t.List[t.Tuple[str, t.Union[str, pa.DataType, "Type"]]],
    ) -> "Type":
        """
        Creates a new `Type` instance representing a struct with the specified fields.

        Args:
            fields: A list of tuples where each tuple contains the name and type of a field.

        Returns:
            A new `Type` instance representing a struct with the specified fields.
        """
        validated_fields = []
        for name, data_type in fields:
            if isinstance(data_type, Type):
                type_ = data_type.value
            elif isinstance(data_type, pa.DataType):
                type_ = data_type
            else:
                type_ = cls._validate_data_type(data_type)
            validated_fields.append(pa.field(name, type_))

        return cls(pa.struct(validated_fields))

    @classmethod
    def from_dict(cls, json_schema: dict):
        """
        Creates a new `Type` instance based on a dictionary representation of the json schema
          of a data type (https://swagger.io/docs/specification/data-models/data-types/).

        Args:
            json_schema: The dictionary representation of the data type, can represent nested values

        Returns:
            A new `Type` instance representing the specified data type.
        """
        type_name = json_schema.get("type")

        if type_name is None:
            msg = "Invalid or missing 'type' key in the schema."
            raise InvalidTypeSchema(msg)

        if type_name == "array":
            items = json_schema.get("items")
            if isinstance(items, dict):
                return cls.list(cls.from_dict(items))
            if isinstance(items, str):
                return cls.list(items)

            msg = "Invalid 'items' type in array schema."
            raise InvalidTypeSchema(msg)

        if type_name == "object":
            properties = json_schema.get("properties")
            if not isinstance(properties, dict):
                msg = "Invalid 'properties' type in object schema."
                raise InvalidTypeSchema(msg)

            fields = [(name, cls.from_dict(prop)) for name, prop in properties.items()]

            return cls.struct(fields)

        if isinstance(type_name, str):
            type_format = json_schema.get("format", None)

            if type_format == "date-time":
                return cls(pa.timestamp("us", tz="UTC"))

            return cls(type_name)

        msg = f"Invalid 'type' value: {type_name}"
        raise InvalidTypeSchema(msg)

    def to_dict(self) -> dict:
        """
        Converts the `Type` instance to its JSON representation.

        Returns:
            A dictionary representing the JSON schema of the data type.
        """
        if isinstance(self.value, pa.ListType):
            items = self.value.value_type
            if isinstance(items, pa.DataType):
                return {"type": "array", "items": Type(items).to_dict()}

        elif isinstance(self.value, pa.StructType):
            fields = [(field.name, Type(field.type).to_dict()) for field in self.value]
            return {"type": "object", "properties": dict(fields)}

        elif isinstance(self.value, pa.TimestampType):
            return {"type": "string", "format": "date-time"}

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


class Field:
    """Class representing a single field or column in a Fondant dataset."""

    def __init__(
        self,
        name: str,
        type: Type = Type("null"),
        location: t.Optional[str] = None,
    ) -> None:
        self.name = name
        self.type = type
        self.location = location

    def __repr__(self):
        """Returns a string representation of the `Type` instance."""
        return f"Field({vars(self)})"

    def __eq__(self, other):
        return vars(self) == vars(other)


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
