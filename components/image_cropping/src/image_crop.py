"""
This file contains the image cropping logic
"""
import io
import typing as t

import numpy as np
from PIL import Image, ImageChops, ImageOps


def most_common(lst: t.Sequence[t.Tuple]) -> t.Tuple:
    """Get the most common element from a list

    Args:
        lst (List): input list

    Returns:
        object: most common element
    """
    return max(set(lst), key=lst.count)


def get_image_borders(image: Image.Image) -> t.Tuple:
    """Get the most common color from the image borders

    Args:
        image (Image.Image): input image

    Returns:
        tuple: most common color
    """
    image = np.array(image)

    # get the colors of the image border
    border_colors = [
        *image[:, 0, :].tolist(),
        *image[:, -1, :].tolist(),
        *image[0, :, :].tolist(),
        *image[-1, :, :].tolist(),
    ]

    # map the colors to RGB tuples
    rgb_border_colors = tuple(map(tuple, border_colors))

    # calculate most common RGB color
    color_common = most_common(rgb_border_colors)

    return tuple(color_common)


def remove_borders(
    image_bytes: bytes, cropping_threshold: int = -30, padding: int = 10
) -> bytes:
    """This method removes borders by checking the overlap between
    a color and the original image. By subtracting these two
    it finds the minimal bounding box.

    Args:
        image_bytes (bytes): input images in bytes
        cropping_threshold (int): threshold parameter used for detecting borders
        padding (int): padding for the image cropping

    Returns:
      bytes: cropped image
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    color_common = get_image_borders(image)
    background = Image.new(image.mode, image.size, color_common)

    diff = ImageChops.difference(image, background)
    diff = ImageChops.add(diff, diff, 1.0, cropping_threshold)

    bbox = diff.getbbox()
    if not bbox:
        return image_bytes

    bbox = (
        np.max([0, bbox[0] - padding]),
        np.max([0, bbox[1] - padding]),
        np.min([width, bbox[2] + padding]),
        np.min([height, bbox[3] + padding]),
    )

    image_crop = image.crop(bbox)

    # if the image is not square, add padding to the shorter side
    # in color `color_common`
    if image_crop.size[0] != image_crop.size[1]:
        if image_crop.size[0] > image_crop.size[1]:
            padding = int((image_crop.size[0] - image_crop.size[1]) / 2)
            image_crop = ImageOps.expand(
                image_crop, border=(0, padding), fill=color_common
            )
        else:
            padding = int((image_crop.size[1] - image_crop.size[0]) / 2)
            image_crop = ImageOps.expand(
                image_crop, border=(padding, 0), fill=color_common
            )

    # serialize image to JPEG
    crop_bytes = io.BytesIO()
    image_crop.save(crop_bytes)

    return crop_bytes.getvalue()
