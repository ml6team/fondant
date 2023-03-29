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

    MIN_HEIGHT = 200
    MIN_WIDTH = 200


@dataclass
class EmbeddingConfig:
    """
    Configs for the embedding component
    Params:
        MODEL_ID (int): HF model id to use for embedding.
        BATCH_SIZE (int): Batch size to use when embedding.
    """

    MODEL_ID = "openai/clip-vit-large-patch14"
    BATCH_SIZE = 10
