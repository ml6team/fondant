import logging

import dask.dataframe as dd
import weaviate
from fondant.component import DaskWriteComponent
from weaviate.util import _capitalize_first_letter

logger = logging.getLogger(__name__)


class EmbeddingsToWeaviate(DaskWriteComponent):
    def __init__(
        self,
        *_,
        weaviate_url: str,
        model: str,
        dataset: str,
        vectorizer: dict,
        overwrite: bool,
    ):
        self.client = weaviate.Client(weaviate_url)
        self.overwrite = overwrite

        # following weaviate convention for upper case class names
        self.class_name = _capitalize_first_letter(f"{model}_{dataset}")
        self.class_obj = {
            "class": self.class_name,
            "properties": [
                {
                    "name": "passage",
                    "dataType": ["text"],
                },
                {  # id of the passage in the passage dataset not to mix up with weaviate's uuid
                    "name": "identifier",
                    "dataType": ["text"],
                },
            ],
        }

        # if vectorizer "none" is provided, one has to compute embeddings before querying from weaviate
        # e.g. computing query embeddings beforehand and running vectorstore.similarity_search_by_vector() in langchain
        self.class_obj.update(vectorizer)

    def write(self, dataframe: dd.DataFrame):
        # using weaviate instead of its langchain wrapper because langchain does not yet provide ingestion of pre-computed embeddings
        if self.overwrite:
            self.client.schema.delete_class(self.class_name)
            self.client.schema.create_class(self.class_obj)
        else:
            if not self.client.schema.exists(self.class_name):
                self.client.schema.create_class(self.class_obj)
            else:
                # do nothing
                return dataframe

        self.client.batch.configure(batch_size=100, dynamic=True, num_workers=2)
        with self.client.batch as batch:
            for part in dataframe.partitions:
                df = part.compute()
                for identifier, data in df.iterrows():
                    properties = {
                        "passage": data["text_data"],
                        "identifier": identifier,
                    }
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        vector=data["text_embedding"],
                    )

        return dataframe
