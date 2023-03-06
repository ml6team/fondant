"""
General class for Blip model
"""
import logging
import sys
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# pylint: disable=import-error, wrong-import-position
sys.path.insert(0, "/src/BLIP")
from BLIP.models.blip import blip_decoder
from helpers.logger import get_logger

logger = get_logger(name=__name__, level=logging.INFO)


# pylint: disable=too-few-public-methods
class BlipModel:
    """Blip Model class"""

    def __init__(self, model_path: str, image_size=384):
        """
        Function that initialize the blip model
        Args:
            model_path (str): the path to the blip model
            image_size (str): the image size to resize the image before captioning
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info('BLIP model initialized with %s device', self.device)
        self.med_config = "./BLIP/configs/med_config.json"
        self.image_size = image_size
        self.model = self._init_model(model_path=model_path, device=self.device)
        # Normalization values taken from
        # https://github.com/salesforce/BLIP/blob/d10be550b2974e17ea72e74edc7948c9e5eab884/predict.py
        self.image_transformer = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def _init_model(self, model_path: str, device: str) -> blip_decoder:
        """
        Initialize the blip model
        Args:
            model_path (str): the path to the blip model
            device (str): the device to initialize the model to
        Returns:
            blip_decoder: the blip model
        """
        model = blip_decoder(pretrained=model_path, image_size=self.image_size, vit='base',
                             med_config=self.med_config)
        model.eval()
        model = model.to(device)

        return model

    def _preprocess_images(self, raw_image: Image) -> torch.Tensor:
        """
        Function that preprocesses the image before captioning
        Args:
            raw_image (Image): the image to caption
        Returns:
            torch.Tensor: a tensor of preprocessed image
        """
        image = self.image_transformer(raw_image).unsqueeze(0)

        return image

    # pylint: disable=too-many-arguments
    def caption_images(self, image_paths: List[str], min_length: int,
                       max_length: int, beams: int) -> List[str]:
        """
        Main function to caption the image
        Args:
            image_paths (List[str]): a list of image paths to caption
            min_length (int): the minimum caption length
            max_length (int): the maximum caption length
            beams (int): parameters to increase the caption per image quality but required more
            compute time
        Returns:
            List[str]: a list of image captions
        """
        image_tensors = []

        with torch.no_grad():
            for image_path in image_paths:
                image = Image.open(image_path).convert('RGB')
                image = self._preprocess_images(image)
                image_tensors.append(image)

            # pylint: disable=no-member
            image_tensors = torch.cat(image_tensors, dim=0).to(self.device)
            captions = self.model.generate(image_tensors,
                                           sample=True,
                                           num_beams=beams,
                                           max_length=max_length,
                                           min_length=min_length)
        return captions
