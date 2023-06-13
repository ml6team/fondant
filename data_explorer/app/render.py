"""Methods for rendering image data in AgGrid tables"""
import base64
import io

from typing import Tuple
from PIL import Image
from st_aggrid.shared import JsCode


def load_image(image_data: bytes,
               size: Tuple[int, int] = (100, 100)
               ) -> str:
    """Load data from image column and convert to encoded png

    Args:
        image_data (bytes): image
        size (Tuple[int, int], optional): size for displaying image. Defaults to (100, 100).

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
    """Format image for rendering in table

    Args:
        image (str): image string

    Returns:
        str: formatted image string
    """
    return f"data:image/png;base64, {image}"


def make_render_image_template(field: str) -> JsCode:
    """Rendering code for Ag Grid image columns

    Args:
        field (str): image field

    Returns:
        JsCode: rendering code for the image field
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
