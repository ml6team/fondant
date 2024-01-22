import typing as t

import dask.dataframe as dd
import pandas as pd
import weaviate
from fondant.component import PandasTransformComponent


class RetrieveFromWeaviateComponent(PandasTransformComponent):
    def __init__(
        self,
        *,
        weaviate_url: str,
        class_name: str,
        top_k: int,
        additional_config: t.Optional[dict],
        additional_headers: t.Optional[dict],
        hybrid_query: t.Optional[str],
        hybrid_alpha: t.Optional[float],
        rerank: bool,
        **kwargs,
    ) -> None:
        """
        Args:
            weaviate_url: An argument passed to the component.
            class_name: Name of class to query
            top_k: Amount of context to return.
            kwargs: Unhandled keyword arguments passed in by Fondant.
            additional_config: Additional configuration passed to the weaviate client.
            additional_headers: Additional headers passed to the weaviate client.
            hybrid_query: The hybrid query to be used for retrieval. Optional parameter.
            hybrid_alpha: Argument to change how much each search affects the results. An alpha
             of 1 is a pure vector search. An alpha of 0 is a pure keyword search.
            rerank: Whether to rerank the results based on the hybrid query. Defaults to False.
             Check this notebook for more information on reranking:
             https://github.com/weaviate/recipes/blob/main/ranking/cohere-ranking/cohere-ranking.ipynb
             https://weaviate.io/developers/weaviate/search/rerank.
        """
        # Initialize your component here based on the arguments
        self.client = weaviate.Client(
            url=weaviate_url,
            additional_config=additional_config if additional_config else None,
            additional_headers=additional_headers if additional_headers else None,
        )
        self.class_name = class_name
        self.k = top_k
        self.hybrid_query, self.hybrid_alpha = self.validate_hybrid_query(
            hybrid_query,
            hybrid_alpha,
        )
        self.rerank = rerank

    @staticmethod
    def validate_hybrid_query(
        hybrid_query: t.Optional[str],
        hybrid_alpha: t.Optional[float],
    ):
        if hybrid_query is not None and hybrid_alpha is None:
            msg = (
                "If hybrid_query is specified, hybrid_alpha must be specified as well."
            )
            raise ValueError(
                msg,
            )

        return hybrid_query, hybrid_alpha

    def validate_reranker(self, dataframe: dd.DataFrame) -> None:
        if self.rerank and "prompt" not in dataframe.columns:
            msg = (
                "If rerank is specified, dataframe must contain a 'text' column. Reranking is"
                " only supported for text data and not for embeddings."
            )
            raise ValueError(
                msg,
            )

    def teardown(self) -> None:
        del self.client

    def retrieve_chunks_from_embeddings(self, vector_query: list):
        """Get results from weaviate database."""
        query = (
            self.client.query.get(self.class_name, ["passage"])
            .with_near_vector({"vector": vector_query})
            .with_limit(self.k)
            .with_additional(["distance"])
        )
        if self.hybrid_query is not None:
            query = query.with_hybrid(query=self.hybrid_query, alpha=self.hybrid_alpha)

        result = query.do()

        result_dict = result["data"]["Get"][self.class_name]
        return [retrieved_chunk["passage"] for retrieved_chunk in result_dict]

    def retrieve_chunks_from_prompts(self, prompt: str):
        """Get results from weaviate database."""
        query = (
            self.client.query.get(self.class_name, ["passage"])
            .with_near_text({"concepts": [prompt]})
            .with_limit(self.k)
        )
        if self.hybrid_query is not None:
            query = query.with_hybrid(query=self.hybrid_query, alpha=self.hybrid_alpha)

        if self.rerank:
            query = query.with_additional(
                'rerank(property: "passage" query: "prompt") { score }',
            )

        result = query.do()

        result_dict = result["data"]["Get"][self.class_name]
        return [retrieved_chunk["passage"] for retrieved_chunk in result_dict]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.validate_reranker(dataframe)

        if "embedding" in dataframe.columns:
            dataframe["retrieved_chunks"] = dataframe["embedding"].apply(
                self.retrieve_chunks_from_embeddings,
            )

        elif "prompt" in dataframe.columns:
            dataframe["retrieved_chunks"] = dataframe["prompt"].apply(
                self.retrieve_chunks_from_prompts,
            )
        else:
            msg = "Dataframe must contain either an 'embedding' column or a 'prompt' column."
            raise ValueError(
                msg,
            )

        return dataframe
