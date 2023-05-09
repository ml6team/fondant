"""This module defines common schemas and datatypes used to define Fondant manifests, components
and pipelines.
"""

import enum
import typing as t

KubeflowCommandArguments = t.List[t.Union[str, t.Dict[str, str]]]


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
    """Class representing a single field or column in a Fondant subset."""

    name: str
    type: Type
