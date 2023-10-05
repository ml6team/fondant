import logging

import dask
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)

dask.config.set(scheduler="processes")


def mask_image(image_data, boxes):
    if len(boxes) > 0:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in boxes:
            x1, y1, x2, y2 = tuple(box)
            # crop image
            cropped_image = np.array(image)[y1:y2, x1:x2]

            if cropped_image.any():
                # compute average color within cropped image
                avg_color_per_row = np.average(cropped_image, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                r, g, b = tuple(avg_color)
                draw.rectangle(((x1, y1), (x2, y2)), fill=(int(r), int(g), int(b)))

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_data = image_bytes.getvalue()

    return image_data


class MaskImagesComponent(PandasTransformComponent):
    """Component that masks images based on bounding boxes, as proposed in
    [T-MARS](https://arxiv.org/abs/2307.03132).
    """

    def __init__(self, *args) -> None:
        pass

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info("Masking images based on boxes...")
        result = dataframe["images"].apply(
            lambda x: mask_image(x.data, x.boxes), axis=1
        )

        dataframe[("images", "data")] = result

        return dataframe
