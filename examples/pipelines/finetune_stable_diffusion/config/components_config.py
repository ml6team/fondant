from dataclasses import dataclass


@dataclass
class LoadFromHubConfig:
    """
    Configs for the dataset loader component
    Params:
        DATASET_NAME (str): Name of the dataset on the hub.
        BATCH_SIZE (int): Batch size to use when creating image metadata.
    """

    DATASET_NAME = "lambdalabs/pokemon-blip-captions"
    BATCH_SIZE = 1000


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


@dataclass
class ClipRetrievalConfig:
    """
    Configs for CLIP image retrieval component
    Params:
        LAION_INDEX_URL(str): url of the indices of the metadata. Those indices need to be
         transformed in case you decide to use only a subset of the dataset
        LAION_METADATA_URL (str): url to the metadata of laion dataset metadata (arrow format). It
         can either contain a subset of the laion 5b metadata (e.g. laion-en) or all of the metadata
        NB_IMAGES_KNN (int): The ratio of number of image to retrieve via the knn strategy
         (per image)
        NB_IMAGES_CENTROID (int): The ratio of number of image to retrieve via the centroid strategy
    """

    LAION_INDEX_URL = "gs://express-sd-datasets/laion-5b/2b-en/image.index/*"
    LAION_METADATA_URL = (
        "gs://express-sd-datasets/laion-5b/metadata/metadata/2B-en.arrow"
    )
    NB_IMAGES_KNN = 500
    NB_IMAGES_CENTROID = 1_000_000
