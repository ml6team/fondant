"""Helper functions for getting field types from a dataframe."""

import typing as t

from fondant.core.schema import Field


def get_fields_by_types(
    fields: t.Dict[str, Field],
    field_types: t.List[str],
) -> t.List[str]:
    filtered_fields = []

    for field, f_type in fields.items():
        if any(ftype in f_type.type.to_json()["type"] for ftype in field_types):
            filtered_fields.append(field)

    return filtered_fields


def get_string_fields(fields: t.Dict[str, Field]) -> t.List[str]:
    return get_fields_by_types(fields, ["string", "utf8"])


def get_image_fields(fields: t.Dict[str, Field]) -> t.List[str]:
    return get_fields_by_types(fields, ["binary"])


def get_numeric_fields(fields: t.Dict[str, Field]) -> t.List[str]:
    numeric_types = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]
    return get_fields_by_types(fields, numeric_types)
