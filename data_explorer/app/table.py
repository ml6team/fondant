"""Logic for constructing and configuring AgGrid tables"""
from typing import Dict, List

import dask.dataframe as dd
from render import image_renderer, load_image, make_render_image_template
from st_aggrid import GridOptionsBuilder


def convert_image_column(dataframe: dd.DataFrame, field: str) -> dd.DataFrame:
    """Add operations for rendering an image column

    Args:
        dataframe (dd.DataFrame): input dataframe
        field (str): image column

    Returns:
        dd.DataFrame: dataframe with formatted image column
    """    
    dataframe[field] = dataframe[field].apply(load_image)
    dataframe[field] = dataframe[field].apply(image_renderer)

    return dataframe


def configure_image_builder(builder: GridOptionsBuilder, field: str):
    """Configure image rendering for AgGrid Table

    Args:
        builder (GridOptionsBuilder): grid option builder
        field (str): image field
    """
    render_template = make_render_image_template(field)
    builder.configure_column(field, field, cellRenderer=render_template)


def get_image_fields(fields: Dict[str, str]) -> List[str]:
    """Get the image fields of the dataframe

    Args:
        fields (Dict[str, str]): dictionary with fields and field types

    Returns:
        List[str]: list of image fields
    """
    # check which of the columns contain byte data
    image_fields = []
    for k, v in fields.items():
        if v == "object":
            image_fields.append(k)
    return image_fields


def get_numeric_fields(fields: Dict[str, str]) -> List[str]:
    """Get the numeric fields of the dataframe

    Args:
        fields (Dict[str, str]): dictionary with fields and field types

    Returns:
        List[str]: list of numeric fields
    """
    # check which of the columns contain byte data
    numeric_fields = []
    for k, v in fields.items():
        if "int" in v or "float" in v:
            numeric_fields.append(k)
    return numeric_fields
