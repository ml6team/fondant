"""Helper functions for getting field types from a dataframe."""

import typing as t


def get_fields_by_types(
    fields: t.Dict[str, str],
    field_types: t.List[str],
) -> t.List[str]:
    return [
        field
        for field, f_type in fields.items()
        if any(ftype in f_type for ftype in field_types)
    ]


def get_string_fields(fields: t.Dict[str, str]) -> t.List[str]:
    return get_fields_by_types(fields, ["string", "utf8"])


def get_image_fields(fields: t.Dict[str, str]) -> t.List[str]:
    return get_fields_by_types(fields, ["binary"])


def get_numeric_fields(fields: t.Dict[str, str]) -> t.List[str]:
    return get_fields_by_types(fields, ["int", "float"])
