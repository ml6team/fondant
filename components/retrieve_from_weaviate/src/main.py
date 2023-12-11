import pandas as pd
import weaviate
from fondant.component import PandasTransformComponent


class RetrieveChunks(PandasTransformComponent):
    def __init__(
        self,
        *,
        weaviate_url: str,
        class_name: str,
        top_k: int,
        **kwargs,
    ) -> None:
        """
        Args:
            weaviate_url: An argument passed to the component.
            class_name: Name of class to query
            top_k: Amount of context to return.
            kwargs: Unhandled keyword arguments passed in by Fondant.
        """
        # Initialize your component here based on the arguments
        self.client = weaviate.Client(weaviate_url)
        self.class_name = class_name
        self.k = top_k

    def retrieve_chunks(self, vector_query: list):
        """Get results from weaviate database."""
        result = (
            self.client.query.get(self.class_name, ["passage"])
            .with_near_vector({"vector": vector_query})
            .with_limit(self.k)
            .with_additional(["distance"])
            .do()
        )
        result_dict = result["data"]["Get"][self.class_name]
        return [retrieved_chunk["passage"] for retrieved_chunk in result_dict]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["retrieved_chunks"] = dataframe["embedding"].apply(
            self.retrieve_chunks,
        )
        return dataframe
