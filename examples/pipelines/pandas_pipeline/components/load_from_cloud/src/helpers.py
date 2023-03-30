"""Helper functionalities for the loader component"""
import os
import io
from PIL import Image

import pandas as pd


def image_to_byte_array(image: Image, image_format: str) -> bytes:
    """
    Function that converts an image to a byte array
    Args:
        image (Image): the image to convert
        image_format (str): the format to convert to
    Returns:
        bytes: Image byte array
    """
    # BytesIO is a file-like buffer stored in memory
    image_byte_array = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(image_byte_array, format=image_format)
    # Turn the BytesIO object back into a bytes object
    image_byte_array = image_byte_array.getvalue()
    return image_byte_array


def create_pd_dataset(images_dir: str, image_format: str = "JPEG") -> pd.DataFrame:
    """
    Function that creates pandas dataset from images stored locally
    Args:
        images_dir (str): the director of the images
        image_format (str): the format to convert the images to before storing in the dataset
    Returns:
        pd.Dataframe: a pandas dataframe of images and their associated metadata
    """
    dataset = []
    index = 0
    for filename in os.listdir(images_dir):
        img_path = os.path.join(images_dir, filename)
        if os.path.isfile(img_path):
            with Image.open(img_path) as img:
                width, height = img.size
                bytestring = image_to_byte_array(img, image_format)
                dataset.append({
                    "index": f"seed_{index}",
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "image": bytestring
                })
                index += 1
    return pd.DataFrame(dataset)
