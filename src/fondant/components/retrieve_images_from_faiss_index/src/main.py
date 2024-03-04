import logging
import typing as t

import dask.dataframe as dd
import faiss
import fsspec
import pandas as pd
import torch
from dask.distributed import Client, get_worker
from dask_cuda import LocalCUDACluster
from fondant.component import DaskTransformComponent
from transformers import AutoTokenizer, CLIPTextModelWithProjection

logger = logging.getLogger(__name__)


class RetrieveImagesFromFaissIndex(DaskTransformComponent):
    """Retrieve images from a faiss index using CLIP embeddings."""

    def __init__(  # PLR0913
        self,
        dataset_path: str,
        faiss_index_path: str,
        clip_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        num_images: int = 2,
    ):
        self.model_id = clip_model
        self.number_of_images = num_images

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download faiss index to local machine
        logger.info(f"Downloading faiss index from {faiss_index_path}")

        with fsspec.open(faiss_index_path, "rb") as f:
            file_contents = f.read()

            with open("faiss_index", "wb") as out:
                out.write(file_contents)

        self.search_index = faiss.read_index("faiss_index")

        dataset = dd.read_parquet(dataset_path)
        if "url" not in dataset.columns:
            msg = "Dataset does not contain column 'url'"
            raise ValueError(msg)
        self.image_urls = dataset["url"].compute().to_list()

    def setup(self) -> Client:
        if self.device == "cuda":
            cluster = LocalCUDACluster()
            return Client(cluster)

        return super().setup()

    def embed_prompt(self, prompt: str):
        worker = get_worker()
        if hasattr(worker, "model"):
            tokenizer = worker.tokenizer
            model = worker.model
        else:
            logger.info("Initializing model '%s' on worker '%s", self.model_id, worker)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id).to(self.device)
            model = CLIPTextModelWithProjection.from_pretrained(self.model_id).to(
                self.device,
            )

            worker.model = model
            worker.tokenizer = tokenizer

        inputs = tokenizer([prompt], padding=True, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.text_embeds.cpu().detach().numpy().astype("float64")

    def retrieve_from_index(
        self,
        query: float,
        number_of_images: int = 2,
    ) -> t.List[str]:
        _, indices = self.search_index.search(query, number_of_images)
        return indices.tolist()[0]

    def transform_partition(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        results = []

        for index, row in dataframe.iterrows():
            if "prompt" in dataframe.columns:
                prompt = row["prompt"]
                query = self.embed_prompt(prompt)

            elif "embedding" in dataframe.columns:
                prompt = None
                query = row["embedding"]
            else:
                msg = (
                    "Dataframe does not contain a prompt or embedding column. "
                    "Please provide one of both."
                )
                raise ValueError(msg)

            indices = self.retrieve_from_index(query, self.number_of_images)
            for i in indices:
                url = self.image_urls[i]
                row_to_add = (index, prompt, i, url) if prompt else (index, i, url)
                results.append(row_to_add)

        results_df = pd.DataFrame(
            results,
            columns=["prompt_id", "prompt", "image_index", "image_url"],
        )
        results_df = results_df.astype({"prompt_id": str})
        return results_df

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        return dataframe.map_partitions(self.transform_partition)
