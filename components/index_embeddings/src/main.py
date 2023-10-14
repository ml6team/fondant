import logging

import dask.dataframe as dd
import weaviate
from fondant.component import DaskWriteComponent
from weaviate.util import _capitalize_first_letter

logger = logging.getLogger(__name__)


class IndexEmbeddingsComponent(DaskWriteComponent):
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

        self.client.batch.configure(batch_size=100, dynamic=True, num_workers=2)
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

        return dataframe
