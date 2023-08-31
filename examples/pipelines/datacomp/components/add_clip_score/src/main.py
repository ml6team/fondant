import logging

import dask
import numpy as np
import pandas as pd

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)

dask.config.set(scheduler="processes")


def compute_clip_score(image_features, text_features):
    image_features = image_features / np.linalg.norm(
        image_features, axis=-1, keepdims=True
    )
    text_features = text_features / np.linalg.norm(
        text_features, axis=-1, keepdims=True
    )
    similarity = text_features @ image_features.T

    return similarity


class AddClipScore(PandasTransformComponent):
    """Component that adds the CLIP score by computing cosine similarity between
    image and text embeddings.
    """

    def __init__(self, *args) -> None:
        pass

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding CLIP score...")

        dataframe[("imagetext", "clipl14score")] = dataframe.apply(
            lambda x: compute_clip_score(x.embeddings.data, x.textembedding.data),
            axis=1,
        )

        return dataframe
