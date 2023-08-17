import io
import os

import pandas as pd
from httpx import Response

from src.main import DownloadImagesComponent


def test_transform(respx_mock):
    """Test the component transform method."""
    # Define input data and arguments
    # These can be parametrized in the future
    ids = [
        "a",
        "b",
        "c",
    ]
    urls = [
        "http://host/path.png",
        "https://host/path.png",
        "https://host/path.jpg",
    ]
    image_size = 256

    # Mock httpx to prevent network calls and return test images
    image_dir = "tests/images"
    images = []
    images = [
        open(os.path.join(image_dir, image), "rb").read() for image in os.listdir(image_dir)  # noqa
    ]
    for url, image in zip(urls, images):
        respx_mock.get(url).mock(return_value=Response(200, content=image))

    component = DownloadImagesComponent(
        timeout=10,
        retries=0,
        image_size=image_size,
        resize_mode="border",
        resize_only_if_bigger=False,
        min_image_size=0,
        max_aspect_ratio=float("inf"),
    )

    input_dataframe = pd.DataFrame(
        {
            ("images", "url"): urls,
        },
        index=pd.Index(ids, name="id"),
    )

    # Use the resizer from the component to generate the expected output images
    # But use the image_size argument to validate actual resizing
    resized_images = [component.resizer(io.BytesIO(image))[0] for image in images]
    expected_dataframe = pd.DataFrame(
        {
            ("images", "data"): resized_images,
            ("images", "width"): [image_size] * len(ids),
            ("images", "height"): [image_size] * len(ids),
        },
        index=pd.Index(ids, name="id"),
    )

    output_dataframe = component.transform(input_dataframe)

    pd.testing.assert_frame_equal(
        left=expected_dataframe,
        right=output_dataframe,
        check_dtype=False,
    )
