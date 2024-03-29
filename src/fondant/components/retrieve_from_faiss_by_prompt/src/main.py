import logging
import os
import typing as t

import dask
import dask.dataframe as dd
import faiss
import fsspec
import pandas as pd
import torch
from dask.distributed import Client, LocalCluster, get_worker
from dask_cuda import LocalCUDACluster
from fondant.component import PandasTransformComponent
from transformers import AutoTokenizer, CLIPTextModelWithProjection

logger = logging.getLogger(__name__)


class RetrieveFromFaissByPrompt(PandasTransformComponent):
    """Retrieve images from a faiss index using CLIP embeddings."""

    def __init__(  # PLR0913
        self,
        url_mapping_path: str,
        faiss_index_path: str,
        clip_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        num_images: int = 2,
    ):
        self.model_id = clip_model
        self.number_of_images = num_images

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download faiss index to local machine
        if not os.path.exists("faiss_index"):
            logger.info(f"Downloading faiss index from {faiss_index_path}")
            with fsspec.open(faiss_index_path, "rb") as f:
                file_contents = f.read()

            with open("faiss_index", "wb") as out:
                out.write(file_contents)

        dataset = dd.read_parquet(url_mapping_path)
        if "url" not in dataset.columns:
            msg = "Dataset does not contain column 'url'"
            raise ValueError(msg)
        self.image_urls = dataset["url"].compute().to_list()

    def setup(self) -> Client:
        """Setup LocalCudaCluster if gpu is available."""
        dask.config.set({"dataframe.convert-string": False})
        dask.config.set({"distributed.worker.daemon": False})

        if self.device == "cuda":
            cluster = LocalCUDACluster()
            return Client(cluster)

        total_memory = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (
            1024**3
        )
        # We need at least 8Gb RAM for the datacomp small faiss index
        # We should consider calculating the memory required for the index based on the faiss
        # index size
        cores_to_utilise = int(total_memory // 8)
        cluster = LocalCluster(
            processes=True,
            n_workers=cores_to_utilise,
            threads_per_worker=1,
            memory_limit="8 GiB",
        )
        return Client(cluster)

    def embed_prompt(self, prompt: str):
        """Embed prompt using CLIP model."""
        worker = get_worker()
        if worker and hasattr(worker, "model"):
            tokenizer = worker.tokenizer
            model = worker.model

        else:
            logger.info("Initializing model '%s' on worker '%s", self.model_id, worker)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = CLIPTextModelWithProjection.from_pretrained(self.model_id).to(
                self.device,
            )

            worker.model = model
            worker.tokenizer = tokenizer

        inputs = tokenizer([prompt], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = model(**inputs)
        return outputs.text_embeds.cpu().detach().numpy().astype("float64")

    def retrieve_from_index(
        self,
        query: float,
        number_of_images: int = 2,
    ) -> t.List[str]:
        """Retrieve images from faiss index."""
        search_index = faiss.read_index("faiss_index")
        _, indices = search_index.search(query, number_of_images)
        return indices.tolist()[0]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Transform partition of dataframe."""
        results = []
        prompts = dataframe["prompt"]

        for prompt in prompts:
            query = self.embed_prompt(prompt)
            indices = self.retrieve_from_index(query, self.number_of_images)
            for i, idx in enumerate(indices):
                url = self.image_urls[idx]
                row_to_add = (f"{prompt}_{i}", prompt, url)
                results.append(row_to_add)

        results_df = pd.DataFrame(
            results,
            columns=["id", "prompt", "image_url"],
        )
        results_df = results_df.set_index("id")
        return results_df
