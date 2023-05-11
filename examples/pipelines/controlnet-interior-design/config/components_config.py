from dataclasses import dataclass


@dataclass
class LaionRetrievalConfig:
    """
    Configs for the LAION retrieval component

    Args:
        NUM_IMAGES (str): Number of images to retrieve for each prompt.
        AESTHETIC_SCORE (int): Ranking score for aesthetic, higher is prettier.
        AESTHETIC_WEIGHT (float): Weight of the aesthetic score, between 0 and 1.
    """

    NUM_IMAGES = 2
    AESTHETIC_SCORE = 9
    AESTHETIC_WEIGHT = 0.5


@dataclass
class DownloadImagesConfig:
    """
    Configs for the download images component

    Args:
        TIMEOUT (int): ...
        RETRIES (int): ...
        IMAGE_SIZE (float): ...
    """

    TIMEOUT = 10
    RETRIES = 0
    IMAGE_SIZE = 512
    RESIZE_MODE = "center_crop"
    RESIZE_ONLY_IF_BIGGER = False
    MIN_IMAGE_SIZE = 512
    MAX_ASPECT_RATIO = 2.5
