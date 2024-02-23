import typing as t

import dask.dataframe as dd
import faiss
import fsspec
import pandas as pd
from fondant.component import PandasTransformComponent
from transformers import AutoTokenizer, CLIPTextModelWithProjection


class RetrieveFromLaionByPrompt(PandasTransformComponent):
    """Retrieve images from the LAION-5B dataset using a prompt."""

    def __init__(
        self,
        index_url: str,
        dataset_url: str,
        clip_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        num_images: int = 2,
    ):
        self.index_url = index_url
        self.model = CLIPTextModelWithProjection.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)
        self.number_of_images = num_images

        # Download faiss index to local machine
        with fsspec.open(index_url, "rb") as f:
            file_contents = f.read()

            with open("faiss_index", "wb") as out:
                out.write(file_contents)

        self.search_index = faiss.read_index("faiss_index")
        self.dataset_url = dataset_url

        dataset_index = dd.read_parquet(dataset_url)
        self.image_urls = dataset_index["url"].compute().to_list()

    def retrieve_from_index(
        self,
        prompt: str,
        number_of_images: int = 2,
    ) -> t.List[str]:
        inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        query = outputs.text_embeds.cpu().detach().numpy().astype("float64")
        _, indices = self.search_index.search(query, number_of_images)
        return indices.tolist()[0]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        results = []
        for index, row in dataframe.iterrows():
            prompt = row["prompt"]
            indices = self.retrieve_from_index(prompt, self.number_of_images)
            for i in indices:
                image_url = self.image_urls[i]
                print(index, i)
                results.append((index, prompt, i, image_url))

        results_df = pd.DataFrame(
            results,
            columns=["prompt_id", "prompt", "image_index", "image_url"],
        )
        results_df = results_df.astype({"prompt_id": str})
        return results_df
