"""Clean cut classifier"""
import json
from typing import Optional

import numpy as np
import torch
from torch import nn


class CCFClassifierModel(nn.Module):
    """The clean cut classifier model"""

    # pylint: disable=invalid-name
    def __init__(self, clip_id: str):
        """
        Initialize clip model
        Args:
            clip_id (str): the id of the clip model used to extract the embeddings
        """
        super(CCFClassifierModel, self).__init__()
        _, clip_prefix = clip_id.split('/')
        if clip_prefix.lower().startswith('clip-vit-large'):
            self.fc = nn.Linear(768, 2)
        else:
            self.fc = nn.Linear(512, 2)

    def forward(self, x):
        """Forward method"""
        return self.fc(x)


class CCFClassifierInference:
    """Wrapper around CCF inference"""

    def __init__(self,
                 weights: str = 'ccf-classifier-v0.1.pt',
                 config: str = 'ccf-classifier-v0.1-config.json',
                 device: Optional[str] = None):
        """
        Class for inference with teh
        Args:
            weights (str): the path to the weights of the model
            config (str): the path to the config weights of the model
            device (str): the device used for inference
        """
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        with open(config) as infile:
            self.config = json.load(infile)

        self.model = CCFClassifierModel(self.config['clip_id'])
        self.model.load_state_dict(torch.load(weights))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.threshold = self.config['threshold']

    def score(self, embeddings: np.array) -> np.array:
        """
        Function that returns the logits of the classifier
        Args:
            embeddings (np.array): the input embedding array
        Returns:
            np.array: the score array
        """
        logits = self.model(torch.Tensor(embeddings).to(self.device))
        return self.softmax(logits).cpu()[:, 1].detach().numpy()

    def predict(self, embeddings: np.array) -> np.array:
        """
        Function that
        Args:
            embeddings (np.array): the input embedding array
        Returns:
            np.array: a boolean array indicating the output of the classifier
        """
        return self.score(embeddings) >= self.threshold
