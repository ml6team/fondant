"""Helper functions for getting field types from a dataframe."""

import typing as t


def get_image_fields(fields: t.Dict[str, str]) -> t.List[str]:
    """Get the image fields of the dataframe.

    Args:
        fields: dictionary with fields and field types

    Returns:
        List of image fields
    """
    # check which of the columns contain byte data
    image_fields = []
    for k, v in fields.items():
        if v == "binary":
            image_fields.append(k)
    return image_fields


def get_numeric_fields(fields: t.Dict[str, str]) -> t.List[str]:
    """Get the numeric fields of the dataframe.

    Args:
        fields: dictionary with fields and field types

    Returns:
         List of numeric fields
    """
    # check which of the columns contain byte data
    numeric_fields = []
    for k, v in fields.items():
        if "int" in v or "float" in v:
            numeric_fields.append(k)
    return numeric_fields
