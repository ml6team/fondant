"""Clip embedding util functions"""
import os

import numpy as np


def get_average_embedding(embedding_dir: str) -> np.array:
    """
    Function that returns the average embedding from a selection of embeddings
    Args:
        embedding_dir (str): the directory where the embeddings are located
    Returns:
        np.array: the average embedding
    """

    embeddings = [np.load(os.path.join(embedding_dir, embedding_file)) for embedding_file in
                  os.listdir(embedding_dir)]
    avg_embedding = np.array(embeddings).mean(axis=0)
    return avg_embedding
