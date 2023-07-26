
from abc import ABC, abstractmethod

import pandas as pd
import pytest
import requests
from caption_images.src.main import CaptionImagesComponent


class AbstractComponentTest(ABC):
    @abstractmethod
    def create_component(self):
        """
        This method should be implemented by concrete test classes
        to create the specific component
        that needs to be tested.
        """
        raise NotImplementedError

    @abstractmethod
    def create_input_data(self):
        """This method should be implemented by concrete test classes
        to create the specific input data.
        """
        raise NotImplementedError

    @abstractmethod
    def create_output_data(self):
        """This method should be implemented by concrete test classes
        to create the specific output data.
        """
        raise NotImplementedError

    def setUp(self):
        """
        This method will be run before each test method.
        Add any common setup steps for your components here.
        """
        self.component = self.create_component()
        self.input_data = self.create_input_data()
        self.expected_output_data = self.create_output_data()

    def tearDown(self):
        """
        This method will be run after each test method.
        Add any common cleanup steps for your components here.
        """

    def test_transform(self):
        """
        Default test for the transform method.
        Tests if the transform method executes without errors.
        """


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

    @pytest.fixture(autouse=True)
    def __setUp(self):
        self.component = self.create_component()
        self.input_data = self.create_input_data()
        self.expected_output_data = self.create_output_data()

    def test_transform(self):
        output = self.component.transform(self.input_data)
        assert output.equals(self.expected_output_data)

    def tearDown(self):
        del self.component
        del self.input_data
        del self.expected_output_data
