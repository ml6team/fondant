"""This module defines common schemas and datatypes used to define Express manifests, components
and pipelines."""

import enum
import types
import typing as t


class Type(enum.Enum):
    """Supported types.

    Based on:
    - https://arrow.apache.org/docs/python/api/datatypes.html#api-types
    - https://pola-rs.github.io/polars/py-polars/html/reference/datatypes.html
    """

    bool: str = "bool"

    int8: str = "int8"
    int16: str = "int16"
    int32: str = "int32"
    int64: str = "int64"

    uint8: str = "uint8"
    uint16: str = "uint16"
    uint32: str = "uint32"
    uint64: str = "uint64"

    float16: str = "float16"
    float32: str = "float32"
    float64: str = "float64"

    decimal: str = "decimal"

    time32: str = "time32"
    time64: str = "time64"
    timestamp: str = "timestamp"

    date32: str = "date32"
    date64: str = "date64"
    duration: str = "duration"

    utf8: str = "utf8"

    binary: str = "binary"

    categorical: str = "categorical"

    list: str = "list"
    struct: str = "struct"


class Field(t.NamedTuple):
    """Class representing a single field or column in an Express subset."""

    name: str
    type: Type


class Subset:
    """
    Class representing an Express subset.

    Args:
        specification: The part of the manifest json representing the subset
        base_path: The base path which the subset location is defined relative to
    """

    def __init__(self, specification: dict, *, base_path: str) -> None:
        self._specification = specification
        self._base_path = base_path

    @property
    def location(self) -> str:
        """The resolved location of the subset"""
        return self._base_path.rstrip("/") + self._specification["location"]

    @property
    def fields(self) -> t.Mapping[str, Field]:
        """The fields of the subset returned as a immutable mapping."""
        return types.MappingProxyType(
            {
                name: Field(name=name, type=field["type"])
                for name, field in self._specification["fields"].items()
            }
        )

    def add_field(self, name: str, type_: Type) -> None:
        if name in self._specification["fields"]:
            raise ValueError("A field with name {name} already exists")

        self._specification["fields"][name] = {"type": type_.value}

    def remove_field(self, name: str) -> None:
        del self._specification["fields"][name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._specification!r}"
