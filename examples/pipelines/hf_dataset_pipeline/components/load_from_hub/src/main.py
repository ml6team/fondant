"""
This component uploads a Hugging Face dataset from the hub to a Google Cloud Storage bucket.
"""
import logging
from typing import Optional, Union, Dict

from datasets import Dataset, load_dataset

from express.components.hf_datasets_components import HFDatasetsLoaderComponent, HFDatasetsDatasetDraft
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def create_metadata(batch):
    images = batch["image"]
    sizes = [image.size for image in images]
    
    # add width and height columns
    batch["width"] = [size[0] for size in sizes]
    batch["height"] = [size[1] for size in sizes]
    
    return batch


class SeedDatasetLoader(HFDatasetsLoaderComponent):
    """Class that inherits from Hugging Face data loading """

    @classmethod
    def load(cls, extra_args: Optional[
        Dict[str, Union[str, int, float, bool]]] = None) -> HFDatasetsDatasetDraft:
        """
        An example function showcasing the data loader component using Express functionalities
        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function (e.g. seed data source)
        Returns:
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output datasets/manifest
        """
        # 1) Create data source
        logger.info("Loading caption dataset from the hub...")
        dataset = load_dataset(extra_args["dataset_name"], split="train")

        # 2) Create an example index
        logger.info("Creating index...")
        index_list = [f"image_{idx}" for idx in range(len(dataset))]
        
        # 3) Create dataset draft from index and data sources
        # We store the index itself also as a HF Dataset
        logger.info("Creating draft...")
        index_dataset = Dataset.from_dict({"index": index_list})
        image_dataset = dataset.remove_columns(["text"]).add_column(name="index", column=index_list)
        image_dataset = image_dataset.map(create_metadata, batched=True)
        text_dataset = dataset.remove_columns(["image"]).add_column(name="index", column=index_list)
        
        data_sources = {"images": image_dataset,
                        "captions": text_dataset}
        dataset_draft = HFDatasetsDatasetDraft(index=index_dataset, data_sources=data_sources)

        return dataset_draft


if __name__ == '__main__':
    SeedDatasetLoader.run()