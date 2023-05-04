"""
This component uploads a Hugging Face dataset from the hub to a Google Cloud Storage bucket.
"""
import logging
from typing import Optional, Union, Dict

import pandas as pd
from datasets import Dataset, load_dataset

from fondant.components.hf_datasets_components import HFDatasetsLoaderComponent, HFDatasetsDatasetDraft
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class SeedDatasetLoader(HFDatasetsLoaderComponent):
    """Class that inherits from Hugging Face data loading """

    @classmethod
    def load(cls, extra_args: Optional[
        Dict[str, Union[str, int, float, bool]]] = None) -> HFDatasetsDatasetDraft:
        """
        An example function showcasing the data loader component using Fondant functionalities
        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function (e.g. seed data source)
        Returns:
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output datasets/manifest
        """
        # 1) Create data source
        logger.info("Loading caption dataset from the hub...")
        caption_dataset = load_dataset(extra_args["dataset_name"], split="train")

        # 2) Create an example index
        logger.info("Creating index...")
        index_list = [f"image_{idx}" for idx in range(len(caption_dataset))]
        caption_dataset = caption_dataset.add_column(name="index", column=index_list)
        
        # 3) Create dataset draft from index and data sources
        # We store the index itself also as a HF Dataset
        logger.info("Creating draft...")
        index_df = pd.DataFrame(index_list, columns=['index'])
        index = Dataset.from_pandas(index_df)
        data_sources = {"captions": caption_dataset}
        dataset_draft = HFDatasetsDatasetDraft(index=index, data_sources=data_sources)

        return dataset_draft


if __name__ == '__main__':
    SeedDatasetLoader.run()