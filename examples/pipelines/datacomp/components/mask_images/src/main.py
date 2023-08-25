import logging
import cv2
import pandas as pd
import numpy as np
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


def mask_text(image_data, bounding_boxes):
    # Decode the binary image data to a numpy array
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Let's mask the text (replace it with the average color of the bounding box)
    for box in bounding_boxes:
        x, y, w, h = box
        region = image[y:y+h, x:x+w]
        average_color = np.mean(region, axis=(0, 1))
        image[y:y+h, x:x+w] = average_color

    # And back to binary image data
    _, encoded_image = cv2.imencode('.jpg', image)
    return encoded_image.tobytes()


class MaskImages(PandasTransformComponent):
    """
    Component that masks text from images using bounding boxes 
    Sets the part of the image in the bounding box to the average color of the bounding box
    """

    def __init__(self, *args, **kwargs):
        pass

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info("Masking images...")

        for index, row in dataframe["images"].iterrows():
            image_data = row['data']
            bounding_boxes = row['bounding_boxes']
            masked_image_data = mask_text(image_data, bounding_boxes)
            dataframe.at[index, 'data'] = masked_image_data
        
        return dataframe