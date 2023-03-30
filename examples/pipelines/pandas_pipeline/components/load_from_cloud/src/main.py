"""
This component creates an Express dataset from images located in a remote storage.
"""

import os
import logging
import importlib
import tempfile
from typing import Optional, Union, Dict

from express.components.pandas_components import PandasLoaderComponent, PandasDatasetDraft
from express.logger import configure_logging
from express.storage_interface import StorageHandlerModule
from helpers import create_pd_dataset

STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()

configure_logging()
logger = logging.getLogger(__name__)


class SeedDatasetLoader(PandasLoaderComponent):
    """Class that inherits from pandas data loading """

    @classmethod
    def load(cls, extra_args: Optional[
        Dict[str, Union[str, int, float, bool]]] = None) -> PandasDatasetDraft:
        """
        An example function showcasing the data loader component using Express functionalities
        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function (e.g. seed data source)
        Returns:
            PandasDatasetDraft: a dataset draft that creates a plan for an output
            datasets/manifest
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1) Download the images locally
            logger.info('created temporary directory %s', tmp_dir)
            logger.info("Downloading images ...")
            image_dir = STORAGE_HANDLER.copy_folder(extra_args["dataset_remote_path"], tmp_dir)

            # 2) Creating dataset
            logger.info("Creating dataset...")
            images = create_pd_dataset(images_dir=image_dir)
            index = images["index"]

            # 3) Create dataset draft from index and data sources
            logger.info("Creating draft...")
            data_sources = {"images": images}
            dataset_draft = PandasDatasetDraft(index=index, data_sources=data_sources)

        return dataset_draft


if __name__ == '__main__':
    SeedDatasetLoader.run()
