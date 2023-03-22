from dataclasses import dataclass

@dataclass
class DatasetLoaderConfig:
    """
    Configs for the dataset loader component
    Params:
        DATASET_NAME (str): Name of the dataset on the hub.
    """
    DATASET_NAME = "lambdalabs/pokemon-blip-captions"


@dataclass
class ImageFilterConfig:
    """
    Configs for the image filter component
    Params:
        MIN_HEIGHT (int): Minimum height for each image.
        MIN_WIDTH (int): Minimum width for each image.
    """
    MIN_HEIGHT = 400
    MIN_WIDTH = 400