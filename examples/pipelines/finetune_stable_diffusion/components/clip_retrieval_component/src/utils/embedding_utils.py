"""Clip embedding util functions"""

from typing import List

import numpy as np


def get_average_embedding(embeddings: List[np.array]) -> np.array:
    """
    Function that returns the average embedding from a list of embeddings

    Args:
        embeddings (List[np.array]): list of embeddings.
    Returns:
        np.array: the average embedding
    """
    initial_embedding = next(iter(embeddings))
    embedding_sum = np.zeros(initial_embedding.shape)
    for embedding in embeddings:
        embedding_sum += embedding

    avg_embedding = embedding_sum.mean(axis=0)
    return avg_embedding
