from dataclasses import dataclass


@dataclass
class RetrieveImagesConfig:
    """
    Configs for the retrieve images component

    Args:
        NUM_IMAGES (str): Number of images to retrieve for each prompt.
        AESTHETIC_SCORE (int): Ranking score for aesthetic, higher is prettier.
        AESTHETIC_WEIGHT (float): Weight of the aesthetic score, between 0 and 1.
    """

    NUM_IMAGES = 2
    AESTHETIC_SCORE = 9
    AESTHETIC_WEIGHT = 0.5
