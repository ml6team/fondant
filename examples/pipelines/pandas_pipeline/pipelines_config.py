"""Dataset creation pipeline config"""

from dataclasses import dataclass


@dataclass
class LoadFromCloudConfig:
    """
    Configs for the dataset loader component
    Params:
        DATASET_REMOTE_PATH (str): The remote path of the dataset
    """
    DATASET_REMOTE_PATH = "gs://soy-audio-379412_datasets/test_seed"


@dataclass
class FilterDatasetConifg:
    """
    Configs for the filter dataset component
    Params:
        MIN_IMAGE_WIDTH (str): The minimum width of the image
        MIN_IMAGE_HEIGHT (str): The minimum height of the image
    """
    MIN_IMAGE_WIDTH = 200
    MIN_IMAGE_HEIGHT = 200
