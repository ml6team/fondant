import logging

import pandas as pd
from fondant.component import PandasTransformComponent
from langchain.embeddings import (
    AlephAlphaAsymmetricSemanticEmbedding,
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)
from retry import retry
from utils import secrets_to_env_vars

logger = logging.getLogger(__name__)


class GenerateEmbeddings(PandasTransformComponent):
    def __init__(
        self,
        *_,
        model: str,
        model_provider: str,
    ):
        self.model = model
        self.model_provider = model_provider

        # to pass embedding model API keys via env vars
        secrets_to_env_vars()

    def get_embedding_model(self, model_provider, model: str):
        # contains a first selection of embedding models
        if model_provider == "aleph_alpha":
            return AlephAlphaAsymmetricSemanticEmbedding(model=model)
        if model_provider == "cohere":
            return CohereEmbeddings(model=model)
        if model_provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=model)
        if model_provider == "openai":
            return OpenAIEmbeddings(model=model)
        msg = f"Unknown provider {model_provider}"
        raise ValueError(msg)

    @retry()  # make sure to keep trying even when api call limit is reached
    def get_embeddings_vectors(self, embedding_model, texts):
        return embedding_model.embed_documents(texts.tolist())

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        embedding_model = self.get_embedding_model(self.model_provider, self.model)
        dataframe[("text", "embedding")] = self.get_embeddings_vectors(
            embedding_model,
            dataframe[("text", "data")],
        )
        return dataframe
