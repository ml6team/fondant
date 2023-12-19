import logging
import typing as t

import dask.dataframe as dd
import weaviate
from fondant.component import DaskWriteComponent
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IndexWeaviateComponent(DaskWriteComponent):
    def __init__(
        self,
        *,
        weaviate_url: str,
        batch_size: int,
        dynamic: bool,
        num_workers: int,
        overwrite: bool,
        class_name: str,
        vectorizer: t.Optional[str],
        **kwargs,
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
            for part in tqdm(
                dataframe.partitions,
                desc="Processing partitions",
                total=dataframe.npartitions,
            ):
                df = part.compute()
                for row in tqdm(df.itertuples(), desc="Processing rows", total=len(df)):
                    properties = {
                        "id_": str(row.Index),
                        "passage": row.text,
                    }
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        vector=row.embedding,
                    )
