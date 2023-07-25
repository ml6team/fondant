import pandas as pd
import pytest
import requests
import torch
from caption_images.src.main import CaptionImagesComponent, caption_image_batch, process_image
from transformers import BlipForConditionalGeneration, BlipProcessor


class TestImageProcessing:

    @pytest.fixture(autouse=True)
    def __setup_teardown(self):
        # Setup code: Create instance of BlipProcessor
        self.model_id = 'Salesforce/blip-image-captioning-base'
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to('cpu')

        self.image_data = {
            'images': {
                'data': [],
            },
        }

        image_urls = ["https://cdn.pixabay.com/photo/2023/06/29/09/52/angkor-thom-8096092_960_720.jpg",
                      "https://cdn.pixabay.com/photo/2023/07/19/18/56/japanese-beetle-8137606_960_720.png"]

        for url in image_urls:
            response = requests.get(url)
            self.image_data['images']['data'].append(response.content)

        self.df = pd.DataFrame(self.image_data)

        # Add teardown code if any

    def test_process_image(self):
        img_byte_arr = self.image_data['images']['data'][0]
        output = process_image(img_byte_arr, processor=self.processor, device='cpu')
        assert isinstance(output, torch.Tensor)

    def test_caption_image_batch(self):
        image_batch = pd.Series(self.df['images']['data']).apply(process_image,
                                                      processor=self.processor,
                                                      device='cpu')

        captions = caption_image_batch(image_batch, model=self.model,
                                       processor=self.processor, max_new_tokens=40)

        assert isinstance(captions, pd.Series)
        assert captions.shape[0] == image_batch.shape[0]

    def test_CaptionImagesComponent_transform(self):
        component = CaptionImagesComponent(model_id=self.model_id, batch_size=5, max_new_tokens=40)
        result_df = component.transform(self.df)

        assert isinstance(result_df, pd.DataFrame)
        assert ('captions', 'text') in result_df.columns
