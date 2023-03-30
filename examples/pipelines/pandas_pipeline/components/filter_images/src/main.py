"""
This component creates an Express dataset from images located in a remote storage.
"""

import logging
import tempfile
from typing import Optional, Union, Dict

from express.components.pandas_components import PandasDataset, PandasDatasetDraft, \
    PandasTransformComponent
from express.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class DatasetFilter(PandasTransformComponent):
    """Class that inherits from pandas data transform """

    @classmethod
    def transform(cls, data: PandasDataset, extra_args: Optional[
        Dict[str, Union[str, int, float, bool]]] = None) -> PandasDatasetDraft:
        """
        An example function showcasing the data transform component using Express functionalities
        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function (e.g. seed data source)
        Returns:
            PandasDatasetDraft: a dataset draft that creates a plan for an output datasets/manifest
        """
        # 1) Get one particular data source from the manifest
        logger.info("Loading caption dataset...")
        image_dataset = data.load(data_source="images", columns=["index", "width", "height"])

        # 2) filter datasource
        filtered_idx = image_dataset[
            (image_dataset['width'] >= extra_args["min_image_width"]) &
            (image_dataset['height'] >= extra_args["min_image_height"])
            ]["index"]

        # 3) Create dataset draft which updates the index
        logger.info("Creating draft...")
        dataset_draft = PandasDatasetDraft(index=filtered_idx,
                                           data_sources=data.manifest.data_sources)

        return dataset_draft


if __name__ == '__main__':
    DatasetFilter.run()
