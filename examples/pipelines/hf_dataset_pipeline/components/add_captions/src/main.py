"""
This component adds captions to a Hugging Face dataset.

Technically, it adds a data source to the manifest.
"""
import logging
from typing import Optional, Union, Dict

from express.components.hf_datasets_components import HFDatasetsTransformComponent, HFDatasetsDataset, HFDatasetsDatasetDraft
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

repo_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(repo_id)
model = BlipForConditionalGeneration.from_pretrained(repo_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def add_captions(examples):
    # get batch of images
    images = examples["image"]

    # prepare images for the model
    encoding = processor(images, return_tensors="pt").to(device)

    # generate captions
    generated_ids = model.generate(**encoding, max_new_tokens=50)
    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    examples["new_caption"] = generated_captions

    return examples


class AddCaptions(HFDatasetsTransformComponent):
    """Class that inherits from Hugging Face data transform"""

    @classmethod
    def transform(cls, data: HFDatasetsDataset, args: Optional[
        Dict[str, Union[str, int, float, bool]]] = None) -> HFDatasetsDatasetDraft:
        """
        An example function showcasing the data transform component using Express functionalities
        
        Args:
            data (HFDatasetsDataset[TIndex, TData]): express dataset providing access to data of a
             given type
            args ([Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function (e.g. seed data source)
        Returns:
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output datasets/manifest
        """
        
        # 1) Get one particular data source from the manifest
        logger.info("Loading image dataset...")
        image_dataset = data.load(data_source="images")
        
        # 2) Create new data source
        logger.info("Adding alternative captions...")
        alternative_caption_dataset = image_dataset.map(add_captions, batched=True, batch_size=2,
                                                          remove_columns=["image", "width", "height"])

        # 3) Create dataset draft which adds a new data source
        logger.info("Creating draft...")
        data_sources = {"alternative_captions": alternative_caption_dataset}
        dataset_draft = HFDatasetsDatasetDraft(data_sources=data_sources, extending_dataset=data)

        return dataset_draft


if __name__ == '__main__':
    AddCaptions.run()