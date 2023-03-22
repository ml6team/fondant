"""
This component adds a data source to the manifest by embedding the images.
"""
import logging
from typing import Optional, Union, Dict

from express.components.hf_datasets_components import HFDatasetsTransformComponent, HFDatasetsDataset, HFDatasetsDatasetDraft
from express.logger import configure_logging

import torch

from transformers import CLIPProcessor, CLIPVisionModelWithProjection

configure_logging()
logger = logging.getLogger(__name__)


@torch.no_grad()
def embed(examples, processor, model):
    images = examples["images"]

    # prepare images for the model
    inputs = processor(images, return_tensors="pt")

    # embed
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    
    batch = {"embeddings": image_embeds}

    return batch


class EmbeddingComponent(HFDatasetsTransformComponent):
    """
    Class that inherits from Hugging Face data transform.
    """

    @classmethod
    def transform(cls, data: HFDatasetsDataset, extra_args: Optional[
        Dict[str, Union[str, int, float, bool]]] = None) -> HFDatasetsDatasetDraft:
        """
        An example function showcasing the data transform component using Express functionalities
        
        Args:
            data (HFDatasetsDataset[TIndex, TData]): express dataset providing access to data of a
             given type
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function
        Returns:
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output manifest
        """
        
        # 1) Get one particular data source from the manifest
        # TODO support streaming
        logger.info("Loading caption dataset...")
        caption_dataset = data.load(data_source="captions", streaming=True)
        
        # 2) Create embedding dataset
        logger.info("Embedding images...")

        processor = CLIPProcessor.from_pretrained(extra_args["model_id"])
        model = CLIPVisionModelWithProjection.from_pretrained(extra_args["model_id"])

        embedded_dataset = caption_dataset.map(embed,
                                               batched=True, batch_size=extra_args["batch_size"],
                                               fn_kwargs=dict(processor=processor, model=model),
                                               remove_columns=["image", "text"])
        
        # 3) Create dataset draft which adds a data source to the manifest
        logger.info("Creating draft...")
        data_sources = {"embeddings": embedded_dataset}
        dataset_draft = HFDatasetsDatasetDraft(data_sources=data_sources, extending_dataset=data)

        return dataset_draft


if __name__ == '__main__':
    EmbeddingComponent.run()