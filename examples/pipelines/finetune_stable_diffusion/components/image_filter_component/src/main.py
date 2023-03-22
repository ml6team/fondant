"""
This component filters images of the dataset based on image size (minimum height and width).

Technically, it updates the index of the manifest.
"""
import logging
from typing import Optional, Union, Dict

from express.components.hf_datasets_components import HFDatasetsTransformComponent, HFDatasetsDataset, HFDatasetsDatasetDraft
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def filter(examples):
    images = examples["image"]

    for image in images:

    
    return -1


class ImageFilterComponent(HFDatasetsTransformComponent):
    """
    Class that inherits from Hugging Face data transform.
    
    Goal is to leverage streaming."""

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
        logger.info("Loading caption dataset...")
        caption_dataset = data.load(data_source="captions")
        
        # 2) Update index by filtering
        logger.info("Filtering dataset...")
        filtered_dataset = caption_dataset.filter(lambda example: example['HEIGHT'] >= extra_args["min_height"] and example['WIDTH'] >= extra_args["min_width"])
        index = filtered_dataset["index"]
        
        # 3) Create dataset draft which updates the index
        logger.info("Creating draft...")
        data_sources = {"captions": caption_dataset}
        dataset_draft = HFDatasetsDatasetDraft(index=index, data_sources=data_sources)

        return dataset_draft


if __name__ == '__main__':
    ImageFilterComponent.run()