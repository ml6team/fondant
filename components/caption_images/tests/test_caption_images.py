import pandas as pd
import requests

from src.main import CaptionImagesComponent


def test_image_caption_component():
    image_urls = [
        "https://cdn.pixabay.com/photo/2023/06/29/09/52/angkor-thom-8096092_1280.jpg",
        "https://cdn.pixabay.com/photo/2023/07/19/18/56/japanese-beetle-8137606_1280.png",
    ]
    input_dataframe = pd.DataFrame(
        {"images": {"data": [requests.get(url).content for url in image_urls]}},
    )

    expected_output_dataframe = pd.DataFrame(
        data={("captions", "text"): {0: "a motorcycle", 1: "a beetle"}},
    )

    component = CaptionImagesComponent(
        model_id="Salesforce/blip-image-captioning-base",
        batch_size=4,
        max_new_tokens=2,
    )

    output_dataframe = component.transform(input_dataframe)

    pd.testing.assert_frame_equal(
        left=expected_output_dataframe,
        right=output_dataframe,
        check_dtype=False,
    )
