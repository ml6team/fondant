from typing import Optional, Dict, Union

from datasets import load_dataset

from express.components.hf_datasets_components import HFDatasetsLoaderComponent, HFDatasetsDatasetDraft
from express.logger import configure_logging


# pylint: disable=too-few-public-methods
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
        configure_logging()
        
        # 1) Create example data source(s)
        caption_dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

        # 2) Create an example index
        index_list = [f"image_{idx}" for idx in range(len(caption_dataset))]
        caption_dataset.add_column(name="index", column=index_list)
        
        # 2.2) Create data_source dictionary
        data_sources = {"captions": caption_dataset}
        
        # 3) Create dataset draft from index and additional data sources
        dataset_draft = HFDatasetsDatasetDraft(index=index_list, data_sources=data_sources)
        return dataset_draft


if __name__ == '__main__':
    SeedDatasetLoader.run()
