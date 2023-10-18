import logging

import dask.dataframe as dd
import weaviate
from fondant.component import DaskWriteComponent
from weaviate.util import _capitalize_first_letter

logger = logging.getLogger(__name__)


class WriteToWeaviateComponent(DaskWriteComponent):
    def __init__(
        self,
        *_,
        weaviate_url: str,
        batch_size: int,
        dynamic: bool,
        num_workers: int,
        overwrite: bool,
        class_name: str,
        vectorizer: dict,
    ):
        self.client = weaviate.Client(weaviate_url)
        self.client.batch.configure(
            batch_size=batch_size,
            dynamic=dynamic,
            num_workers=num_workers,
        )

        self.overwrite = overwrite

        self.class_name = _capitalize_first_letter(class_name)
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

        self.class_obj.update(vectorizer)

    def write(self, dataframe: dd.DataFrame):
        # using weaviate instead of its langchain wrapper
        # because langchain does not yet provide ingestion of pre-computed embeddings
        if self.overwrite:
            self.client.schema.delete_class(self.class_name)
            self.client.schema.create_class(self.class_obj)
        elif not self.client.schema.exists(self.class_name):
            self.client.schema.create_class(self.class_obj)
        else:
            # do not index
            return dataframe

        with self.client.batch as batch:
            for part in dataframe.partitions:
                dataframe = part.compute()
                for identifier, data in dataframe.iterrows():
                    properties = {
                        "passage": data["text_data"],
                        "identifier": identifier,
                    }
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        vector=data["text_embedding"],
                    )
            return None
