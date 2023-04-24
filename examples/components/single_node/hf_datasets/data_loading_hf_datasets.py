from typing import Optional, Dict, Union

import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

from fondant.components.hf_datasets_components import HFDatasetsLoaderComponent, HFDatasetsDatasetDraft
from fondant.logger import configure_logging


# pylint: disable=too-few-public-methods
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
        configure_logging()
        
        # 1) Create example data source(s)
        caption_dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

        # 2) Create an example index
        index_list = [f"image_{idx}" for idx in range(len(caption_dataset))]
        index_df = pd.DataFrame(index_list, columns=['index'])
        index = Dataset.from_pandas(index_df)

        caption_dataset = concatenate_datasets([index, caption_dataset])
        
        # 2.2) Create data_source dictionary
        data_sources = {"captions": caption_dataset}
        
        # 3) Create dataset draft from index and additional data sources
        dataset_draft = HFDatasetsDatasetDraft(index=index, data_sources=data_sources)
        return dataset_draft


if __name__ == '__main__':
    SeedDatasetLoader.run()
