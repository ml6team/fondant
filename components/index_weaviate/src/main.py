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
        additional_config: t.Optional[dict],
        additional_headers: t.Optional[dict],
        vectorizer: t.Optional[str],
        module_config: t.Optional[dict],
        **kwargs,
    ):
        self.client = weaviate.Client(
            url=weaviate_url,
            additional_config=additional_config if additional_config else None,
            additional_headers=additional_headers if additional_headers else None,
        )

        self.client.batch.configure(
            batch_size=batch_size,
            dynamic=dynamic,
            num_workers=num_workers,
        )

        self.class_name = class_name
        self.vectorizer = vectorizer
        self.module_config = module_config

        if overwrite:
            self.client.schema.delete_class(self.class_name)

        if not self.client.schema.exists(self.class_name):
            class_schema = self.create_class_schema()
            self.client.schema.create_class(class_schema)

    def validate_component(self, dataframe: dd.DataFrame) -> None:
        if "embedding" not in dataframe.columns and self.vectorizer is None:
            msg = "If vectorizer is not specified, dataframe must contain an 'embedding' column."
            raise ValueError(
                msg,
            )

        if self.vectorizer is not None and not self.module_config:
            msg = "If vectorizer is specified, module_config must be specified as well."
            raise ValueError(
                msg,
            )

    def create_class_schema(self) -> t.Dict[str, t.Any]:
        class_schema: t.Dict[str, t.Any] = {
            "class": self.class_name,
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
        }

        if self.vectorizer is not None:
            class_schema["vectorizer"] = self.vectorizer

        if self.module_config is not None:
            class_schema["moduleConfig"] = self.module_config

        return class_schema

    def teardown(self) -> None:
        del self.client

    def write(self, dataframe: dd.DataFrame) -> None:
        self.validate_component(dataframe)

        with self.client.batch as batch:
            for part in tqdm(
                dataframe.partitions,
                desc="Processing partitions",
                total=dataframe.npartitions,
            ):
                df = part.compute()

                for row in df.itertuples():
                    properties = {
                        "id_": str(row.Index),
                        "passage": row.text,
                    }

                    if self.vectorizer is None:
                        batch.add_data_object(
                            data_object=properties,
                            class_name=self.class_name,
                            vector=row.embedding,
                        )
                    else:
                        batch.add_data_object(
                            data_object=properties,
                            class_name=self.class_name,
                        )
