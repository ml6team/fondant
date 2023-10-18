import logging
import typing as t

import dask.dataframe as dd
import weaviate
from fondant.component import DaskWriteComponent

logger = logging.getLogger(__name__)


class IndexWeaviateComponent(DaskWriteComponent):
    def __init__(
        self,
        *_,
        weaviate_url: str,
        batch_size: int,
        dynamic: bool,
        num_workers: int,
        overwrite: bool,
        class_name: str,
        vectorizer: t.Optional[str],
    ):
        self.client = weaviate.Client(weaviate_url)

        self.client.batch.configure(
            batch_size=batch_size,
            dynamic=dynamic,
            num_workers=num_workers,
        )

        self.class_name = class_name

        if overwrite:
            self.client.schema.delete_class(self.class_name)
        if not self.client.schema.exists(self.class_name):
            self.client.schema.create_class(
                {
                    "class": class_name,
                    "properties": [
                        {
                            "name": "passage",
                            "dataType": ["text"],
                        },
                        {  # id of the passage in the passage dataset
                            # not to mix up with weaviate's uuid
                            "name": "id_",
                            "dataType": ["text"],
                        },
                    ],
                    "vectorizer": vectorizer,
                },
            )

    def write(self, dataframe: dd.DataFrame) -> None:
        with self.client.batch as batch:
            for part in dataframe.partitions:
                df = part.compute()
                for row in df.itertuples():
                    properties = {
                        "id_": str(row.Index),
                        "passage": row.text_data,
                    }
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        vector=row.text_embedding,
                    )
