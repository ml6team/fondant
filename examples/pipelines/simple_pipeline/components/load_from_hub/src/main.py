"""
This component loads a seed dataset from the hub.
"""
import io
import logging

import dask.dataframe as dd
import dask.array as da
import numpy as np
from PIL import Image

from fondant.component import FondantLoadComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def extract_width(image_bytes):
    # Decode image bytes to PIL Image object
    pil_image = Image.open(io.BytesIO(image_bytes))
    width = pil_image.size[0]

    return np.int16(width)


def extract_height(image_bytes):
    # Decode image bytes to PIL Image object
    pil_image = Image.open(io.BytesIO(image_bytes))
    height = pil_image.size[1]

    return np.int16(height)


class LoadFromHubComponent(FondantLoadComponent):

    def load(self, *, dataset_name: str):
        """
        Args:
            dataset_name (`str`):
                Name of dataset on the hub.
        
        Returns:
            Dask dataframe
        """
        # 1) Load data, read as Dask dataframe
        logger.info("Loading dataset from the hub...")
        dask_df = dd.read_parquet(f"hf://datasets/{dataset_name}")

        # 2) Add index to the dataframe
        logger.info("Creating index...")
        index_list = [idx for idx in range(len(dask_df))]
        source_list = ["hub" for _ in range(len(dask_df))]

        dask_df["id"] = da.array(index_list)
        dask_df["source"] = da.array(source_list)

        # 3) Rename columns
        dask_df = dask_df.rename(columns={"image": "images_data", "text": "captions_data"})

        # 4) Make sure images are bytes instead of dicts
        dask_df["images_data"] = dask_df["images_data"].map(lambda x: x["bytes"],
                                                            meta=("bytes", bytes))

        # 5) Add width and height columns
        dask_df['images_width'] = dask_df['images_data'].map(extract_width,
                                                             meta=("images_width", int))
        dask_df['images_height'] = dask_df['images_data'].map(extract_height,
                                                              meta=("images_height", int))

        return dask_df


if __name__ == "__main__":
    component = LoadFromHubComponent()
    component.run()
