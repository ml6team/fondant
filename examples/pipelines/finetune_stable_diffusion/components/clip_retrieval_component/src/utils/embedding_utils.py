"""Clip embedding util functions"""

from typing import List

from datasets import Dataset

import numpy as np


def get_average_embedding(embeddings_dataset: Dataset) -> np.array:
    """
    Function that returns the average embedding from a list of embeddings

    Args:
        embeddings_dataset (datasets.Dataset): list of embeddings.
    Returns:
        np.array: the average embedding
    """
    embeddings = [np.array(embedding) for embedding in embeddings_dataset["embeddings"]]
    
    avg_embedding = np.mean(embeddings, axis=0)
    
    return avg_embedding
