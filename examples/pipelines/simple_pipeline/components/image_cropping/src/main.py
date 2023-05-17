"""
This component filters images of the dataset based on image size (minimum height and width).
"""
import io
import logging

import dask
import dask.dataframe as dd
from PIL import Image

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)



def crop(image):
    return image
    # load bytes into image
    # pil_image = Image.open(io.BytesIO(image))
    # pil_image = pil_image.crop((0, 0, 224, 224))
    # # serialize image
    # with io.BytesIO() as output:
    #     pil_image.save(output, format="JPEG")
    #     cropped_image = output.getvalue()
    # return cropped_image

class ImageCroppingComponent(TransformComponent):
    """
    Component that crops images
    """

    def transform(
        self, *, dataframe: dd.DataFrame
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            images: images

        Returns:
            dataset
        """
        dataframe["images_crop"] = dataframe["images_data"].map(crop, meta=("images_crop", "bytes"))
        # dataframe['']
        return dataframe


if __name__ == "__main__":
    component = ImageCroppingComponent.from_file()
    component.run()
