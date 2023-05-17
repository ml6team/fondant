"""This module defines common schemas and datatypes used to define Fondant manifests, components
and pipelines.
"""

import enum
import typing as t

import pyarrow as pa

KubeflowCommandArguments = t.List[t.Union[str, t.Dict[str, str]]]


class Type(enum.Enum):
    """Supported types.

    Based on:
    - https://arrow.apache.org/docs/python/api/datatypes.html#api-types
    - https://pola-rs.github.io/polars/py-polars/html/reference/datatypes.html
    """

    bool = pa.bool_()

    int8 = pa.int8()
    int16 = pa.int16()
    int32 = pa.int32()
    int64 = pa.int64()

    uint8 = pa.uint8()
    uint16 = pa.uint16()
    uint32 = pa.uint32()
    uint64 = pa.uint64()

    float16 = pa.float16()
    float32 = pa.float32()
    float64 = pa.float64()

    decimal = pa.decimal128(38)

    time32 = pa.time32("s")
    time64 = pa.time64("us")
    timestamp = pa.timestamp("us")

    date32 = pa.date32()
    date64 = pa.date64()
    duration = pa.duration("us")

    string = pa.string()
    utf8 = pa.utf8()

    binary = pa.binary()

    bool_list = pa.bool_()

    int8_list = pa.list_(pa.int8())
    int16_list = pa.list_(pa.int16())
    int32_list = pa.list_(pa.int32())
    int64_list = pa.list_(pa.int64())

    uint8_list = pa.list_(pa.uint8())
    uint16_list = pa.list_(pa.uint16())
    uint32_list = pa.list_(pa.uint32())
    uint64_list = pa.list_(pa.uint64())

    float16_list = pa.list_(pa.float16())
    float32_list = pa.list_(pa.float32())
    float64_list = pa.list_(pa.float64())

    decimal_list = pa.list_(pa.decimal128(38))

    time32_list = pa.list_(pa.time32("s"))
    time64_list = pa.list_(pa.time64("us"))
    timestamp_list = pa.list_(pa.timestamp("us"))

    date32_list = pa.list_(pa.date32())
    date64_list = pa.list_(pa.date64())
    duration_list = pa.list_(pa.duration("us"))

    string_list = pa.list_(pa.string())
    utf8_list = pa.list_(pa.utf8())

    binary_list = pa.list_(pa.binary())


class Field(t.NamedTuple):
    """Class representing a single field or column in a Fondant subset."""

    name: str
    type: Type


Field("a", Type.string_list)

a = 2
