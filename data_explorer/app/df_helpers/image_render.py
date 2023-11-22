"""Methods for rendering image data in AgGrid tables."""
import base64
import io
import typing as t

import dask.dataframe as dd
from PIL import Image
from st_aggrid import GridOptionsBuilder
from st_aggrid.shared import JsCode


def load_image(
    image_data: bytes,
    size: t.Optional[t.Tuple[int, int]] = (100, 100),
) -> str:
    """Load data from image column and convert to encoded png.

    Args:
        image_data:: binary data from image column
        size: size for displaying image

    Returns:
        str: encoded image
    """
    image = Image.open(io.BytesIO(image_data)).resize(size)
    # write to PNG
    image = image.convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    # decode to base64
    image_bytes = image_bytes.read()
    image_bytes = base64.b64encode(image_bytes).decode()

    return image_bytes


def image_renderer(image: str) -> str:
    """Format image for rendering in table.

    Args:
        image: image as a base64 encoded string

    Returns:
        Formatted image string
    """
    return f"data:image/png;base64, {image}"


def make_render_image_template(field: str) -> JsCode:
    """Rendering code for Ag Grid image columns.

    Args:
        field: image field

    Returns:
        Rendering code for the image field
    """
    string = f"""
        class ThumbnailRenderer {{
            init(params) {{
                this.eGui = document.createElement('img');
                this.eGui.setAttribute('src', params.data.{field});
                this.eGui.setAttribute('width', '100');
                this.eGui.setAttribute('height', '100');
            }}
            getGui() {{
                return this.eGui;
            }}
        }}"""
    return JsCode(string)


def convert_image_column(dataframe: dd.DataFrame, field: str) -> dd.DataFrame:
    """Add operations for rendering an image column.

    Args:
        dataframe: input dataframe
        field: image column

    Returns:
        Dataframe with formatted image column
    """
    dataframe[field] = dataframe[field].apply(load_image)
    dataframe[field] = dataframe[field].apply(image_renderer)

    return dataframe


def configure_image_builder(builder: GridOptionsBuilder, image_field: str):
    """Configure image rendering for AgGrid Table.

    Args:
        builder: grid option builder
        image_field: name of the image field
    """
    render_template = make_render_image_template(image_field)
    builder.configure_column(image_field, image_field, cellRenderer=render_template)
