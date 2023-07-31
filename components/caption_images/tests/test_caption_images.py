import pandas as pd
import requests
from caption_images.src.main import CaptionImagesComponent
from fondant.abstract_component_test import AbstractComponentTest


class TestCaptionImagesComponent(AbstractComponentTest):
    def create_component(self):
        return CaptionImagesComponent(
            model_id="Salesforce/blip-image-captioning-base",
            batch_size=4,
            max_new_tokens=2,
        )

    def create_input_data(self):
        image_urls = [
            "https://cdn.pixabay.com/photo/2023/06/29/09/52/angkor-thom-8096092_1280.jpg",
            "https://cdn.pixabay.com/photo/2023/07/19/18/56/japanese-beetle-8137606_1280.png",
        ]
        return pd.DataFrame(
            {"images": {"data": [requests.get(url).content for url in image_urls]}},
        )

    def create_output_data(self):
        return pd.DataFrame(
            data={("captions", "text"): {0: "a motorcycle", 1: "a beetle"}},
        )
