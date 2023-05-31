"""
This component processes images with a controlnet aux processor
"""

import logging

import dask.dataframe as dd
from PIL import Image

from processor import Processor
from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ControlnetAuxComponent(TransformComponent):
    """
    Component that processes images with controlnet aux processors
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
        processor_id: str,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            processor_id: processor id

        Returns:
            Dask dataframe
        """
        # load processor
        processor = Processor(processor_id=processor_id)

        dataframe = dataframe.sample(frac=0.2)
        print(f"{len(dataframe)} LENGTH")
        # calculate processed image
        dataframe['conditioning_data'] = dataframe['images_data'].apply(processor,
                                                                        meta=('conditioning_data', 'bytes'))

        return dataframe


if __name__ == "__main__":
    component = ControlnetAuxComponent.from_file()
    component.run()
