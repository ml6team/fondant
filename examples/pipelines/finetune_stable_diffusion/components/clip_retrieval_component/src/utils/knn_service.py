"""Clip retrieval script"""
import json
import io
import base64
import warnings
import logging
from typing import Dict, List

# pylint: disable=import-error
from PIL import Image
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions

from helpers.logger import get_logger
from utils.timer import CatchTime

LOGGER = get_logger(name=__name__, level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)


class ClipRetrievalLaion5B:
    """Class that initializes and starts a clip retrieval job hosted locally using the laion5B
    dataset (either full dataset or subset)"""

    def __init__(self, laion_index_path: str, laion_index_folder: str):
        """
        Parameter initialization
        Args:
            laion_index_path (str): the path to the laion json index path containing job run specs
            laion_index_folder (str): the path to the laion index folder. This folder must contain
            two subfolders:
                -'metadata': contains the laion dataset metadata (arrow format). This folder can
                 either contain a subset of the laion 5b metadata (e.g. laion-en) or all of the
                  metadata
                -'image.index' contains the indices of the metadata. Those indices need to be
                transformed in case you decide to use only a subset of the dataset (more info here:
                https://github.com/rom1504/clip-retrieval/tree/
                a206f7908ed213030e950655335940a77af1e4ea/clip_retrieval/clip_back_prepro)
        """
        self.laion_index_path = laion_index_path
        self.laion_index_folder = laion_index_folder
        self.timer = CatchTime()

    def _write_laion_index_path(self):
        """Write the json index path containing the job run spec"""
        laion_job_dict = {
            "laion5B": {
                "indice_folder": self.laion_index_folder,
                "provide_safety_model": False,
                "enable_faiss_memory_mapping": True,
                "use_arrow": True,
                "enable_hdf5": False,
                "reorder_metadata_by_ivf_index": False,
                "columns_to_return": ["id", "url"],
                "clip_model": "ViT-L/14",
                "enable_mclip_option": False,
            }
        }

        # Serializing json
        json_object = json.dumps(laion_job_dict, indent=4)

        # Writing to indices_paths.json
        with open(self.laion_index_path, "w") as outfile:
            outfile.write(json_object)

    def setup_knn_service(self) -> KnnService:
        """
        Function that sets up the knn service
        Returns:
            KNNService: the knn service client
        """
        self._write_laion_index_path()
        # Load clip indices with helper method
        clip_resources = load_clip_indices(
            indices_paths=self.laion_index_path,
            clip_options=ClipOptions(
                indice_folder="",
                clip_model="ViT-L/14",
                enable_hdf5=False,
                enable_faiss_memory_mapping=True,
                columns_to_return=None,
                reorder_metadata_by_ivf_index=False,
                enable_mclip_option=True,
                use_jit=True,
                use_arrow=False,
                provide_safety_model=False,
                provide_violence_detector=False,
                provide_aesthetic_embeddings=True,
            ),
        )
        # Setup KNN service object with clip resources
        knn_service = KnnService(clip_resources=clip_resources)
        return knn_service

    # pylint: disable=too-many-arguments
    def run_query(
        self,
        nb_images_request: int,
        query: Dict[str, str],
        knn_service: KnnService,
        deduplicate: bool = True,
        benchmark: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Function that runs query on the KNN faiss backend
        Args:
            nb_images_request (int): the number of image to request from the KNN service
            query (Dict[str, str]): the query to request from the KNN service
            knn_service (KNNService): the KNN service client
            deduplicate (bool): whether to deduplicate based on embeddings
            benchmark (bool): enable to benchmark knn service
        Returns:
            List[Dict[str, str]]: the query results
        """
        LOGGER.info("Running query")

        if benchmark is False:
            self.timer.disable()
        else:
            self.timer.enable()
            self.timer.reset()

        self.timer(f"Query for {nb_images_request} samples")

        with self.timer:
            query_inputs = {
                "modality": "image",
                "indice_name": "laion5B",
                "num_images": int(nb_images_request),
                "deduplicate": deduplicate,
                "num_result_ids": int(nb_images_request),
            }

            if "image_file" in query:
                image = Image.open(query["image_file"])
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue())
                query_inputs["image_input"] = img_str
            elif "image_url" in query:
                query_inputs["image_url_input"] = query["image_url"]
            elif "text_query" in query:
                query_inputs["text_input"] = query["text_query"]
            elif "embedding_query" in query:
                query_inputs["embedding_input"] = query["embedding_query"]
            else:
                raise Exception("No text, image or embedding query found")

            results = knn_service.query(**query_inputs)
            LOGGER.info("Query job complete")
            LOGGER.info("A total of %s results were returned", len(results))
            return results

    @staticmethod
    def clip_results_producer(clip_results: List[Dict[str, str]]) -> tuple:
        """
        Laion results producer
        Args:
            clip_results (List[Dict[str, str]]): the clip retrieval results
        Returns:
            tuple: clip retrieval results
        """
        for result in clip_results:
            yield result["id"], result["url"]
