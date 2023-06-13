import base64
import io

from PIL import Image
from st_aggrid.shared import JsCode


def load_image(image_data, size=(100, 100)):
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


def image_renderer(image):
    return f"data:image/png;base64, {image}"


def make_render_image_template(field):
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
