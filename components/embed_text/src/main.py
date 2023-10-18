import logging

import pandas as pd
from fondant.component import PandasTransformComponent
from langchain.embeddings import (
    AlephAlphaAsymmetricSemanticEmbedding,
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)
from langchain.schema.embeddings import Embeddings
from retry import retry
from utils import to_env_vars

logger = logging.getLogger(__name__)


class EmbedTextComponent(PandasTransformComponent):
    def __init__(
        self,
        *_,
        model_provider: str,
        model: str,
        api_keys: dict,
    ):
        self.embedding_model = self.get_embedding_model(model_provider, model)
        to_env_vars(api_keys)

    @staticmethod
    def get_embedding_model(model_provider, model: str) -> Embeddings:
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
    def get_embeddings_vectors(self, texts):
        return self.embedding_model.embed_documents(texts.tolist())

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[("text", "embedding")] = self.get_embeddings_vectors(
            dataframe[("text", "data")],
        )
        return dataframe
