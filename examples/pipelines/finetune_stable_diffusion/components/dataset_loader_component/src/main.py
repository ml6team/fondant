"""
This component loads a seed dataset from the cloud and creates the initial manifest.
"""
from typing import Optional, Union, Dict

from datasets import load_from_disk

from express.components.hf_datasets_components import HFDatasetsLoaderComponent, HFDatasetsDatasetDraft
from express.logger import configure_logging


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
            HFDatasetsDatasetDraft: a dataset draft that creates a plan for an output manifest
        """
        configure_logging()
        
        # 1) Create data source
        storage_options={"project": extra_args["project_id"]}
        dataset.save_to_disk("s3://my-private-datasets/imdb/train", storage_options=storage_options)

        print(len(dataset))

        return dataset


if __name__ == '__main__':
    SeedDatasetLoader.run()