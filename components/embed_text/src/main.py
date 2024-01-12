import logging
import os

import google.cloud.aiplatform as aip
import pandas as pd
from fondant.component import PandasTransformComponent
from langchain.embeddings import (
    AlephAlphaAsymmetricSemanticEmbedding,
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    VertexAIEmbeddings,
)
from langchain.schema.embeddings import Embeddings

logger = logging.getLogger(__name__)


def to_env_vars(api_keys: dict):
    for key, value in api_keys.items():
        os.environ[key] = value


class EmbedTextComponent(PandasTransformComponent):
    def __init__(
        self,
        *,
        model_provider: str,
        model: str,
        api_keys: dict,
        auth_kwargs: dict,
        **kwargs,
    ):
        to_env_vars(api_keys)

        self.embedding_model = self.get_embedding_model(
            model_provider,
            model,
            auth_kwargs,
        )

    @staticmethod
    def get_embedding_model(
        model_provider,
        model: str,
        auth_kwargs: dict,
    ) -> Embeddings:
        if model_provider == "vertexai":
            aip.init(**auth_kwargs)
            return VertexAIEmbeddings(model=model)
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

    # make sure to keep trying even when api call limit is reached
    def get_embeddings_vectors(self, texts):
        return self.embedding_model.embed_documents(texts.tolist())

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["embedding"] = self.get_embeddings_vectors(
            dataframe["text"],
        )
        return dataframe
