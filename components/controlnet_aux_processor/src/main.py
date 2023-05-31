"""
This component processes images with a controlnet aux processor
"""

import logging

import dask.dataframe as dd
from PIL import Image
from controlnet_aux.processor import Processor

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
            processor_id: processor id for the controlnet_aux processor class

        Returns:
            Dask dataframe
        """
        processor_params = {'to_pil': False}

        # load processor
        processor = Processor(processor_id=processor_id,
                              params=processor_params)

        # calculate processed image
        dataframe['conditioning_data'] = dataframe['images_data'].apply(processor,
                                                                        meta=('conditioning_data', 'bytes'))

        return dataframe


if __name__ == "__main__":
    component = ControlnetAuxComponent.from_file()
    component.run()
