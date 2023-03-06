"""Image embedding"""
import os
import logging
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pyarrow.dataset import Scanner

# pylint: disable=import-error
import clip
from helpers import storage_helpers, io_helpers
from helpers.logger import get_logger

LOGGER = get_logger(name=__name__, level=logging.INFO)

# Set to allow pillow to process images of different sizes
Image.MAX_IMAGE_PIXELS = None


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class KfpPipelineImageEmbedder:
    """kfp image embedder """

    # pylint: disable=too-many-arguments
    def __init__(self, parquet_dataset: Scanner, embedding_blob_path: str,
                 tmp_img_path: str, tmp_embedding_path: str, download_list_file_path: str):
        """
        Class that structures the kfp images conversion loop
        Args:
            embedding_blob_path (str): the blob path where the image embeddings will be stored
            parquet_dataset (Scanner): the scanned parquet dataset
            tmp_img_path (str) the temporary path used to store the downloaded images
            tmp_embedding_path (str): the temporary path to save the image embeddings
            download_list_file_path (str): path to file list containing one-per-line list of GCS
             URIs to download
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info('CLIP model initialized with %s device', self.device)
        self.clip_model_vit, self.clip_preprocess_vit = clip.load('ViT-L-14',
                                                                  device=self.device)
        self.parquet_dataset = parquet_dataset
        self.embedding_blob_path = embedding_blob_path
        self.tmp_img_path = tmp_img_path
        self.tmp_embedding_path = tmp_embedding_path
        self.download_list_file_path = download_list_file_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _write_gcs_file_lists(self):
        """
        Function that writes the gcs download and upload list files for bulk download/upload
        """
        with open(self.download_list_file_path, "w") as download_list_file:
            for batch in self.parquet_dataset.to_batches():
                for row in batch.to_pylist():
                    file_uri = row['file_uri']
                    download_list_file.write(file_uri + "\n")

        LOGGER.info("GCS download file list written to %s", self.download_list_file_path)

    def _download_images_to_embed(self):
        """Function that download the images to caption locally"""
        LOGGER.info("Downloading images to caption")
        storage_helpers.copy_files_bulk(self.download_list_file_path,
                                        self.tmp_img_path)
        LOGGER.info("The images to be captioned were successfully downloaded to %s",
                    self.tmp_img_path)

    def _write_embeddings(self, embeddings: np.array, file_ids: List[str]):
        """
        Function that saves the image embeddings as a numpy array
        Args:
            embeddings (np.array): the embeddings array
            file_ids (List[str]): the list of file ids associated with the embeddings
        """
        for embedding_array, file_id in zip(embeddings, file_ids):
            np.save(os.path.join(self.tmp_embedding_path, f"{file_id}.npy"), embedding_array)

    def _embed_images(self):
        """
        Function that embeds the images with CLIP
        """
        LOGGER.info("Starting image embedding with CLIP")

        for batch in self.parquet_dataset.to_batches():
            # pyarrow's batch_size approximates the number of batches to return
            # Batches may be smaller if there aren’t enough rows in the file and can return zero
            # in some occasions
            if len(batch) > 0:
                img_list, file_ids = [], []
                for row in tqdm(batch.to_pylist()):
                    file_uri, file_id = row['file_uri'], row['file_id']
                    file_name = io_helpers.get_file_name(file_uri, return_extension=True)
                    local_file_path = os.path.join(self.tmp_img_path, file_name)
                    img_list.append(self.clip_preprocess_vit(Image.open(local_file_path)))
                    file_ids.append(file_id)
                # pylint: disable=no-member
                img_stack = torch.tensor(np.stack(img_list), device=self.device)
                embeddings = self.clip_model_vit.encode_image(img_stack).cpu().detach().numpy()
                self._write_embeddings(embeddings, file_ids)

        LOGGER.info("Image embedding completed")

    def _upload_image_embeddings(self):
        """Function that uploads the image embeddings from local disk to gcs"""
        storage_helpers.copy_folder_bulk(self.tmp_embedding_path, self.embedding_blob_path)

    def start(self):
        """
        Function that starts the image embedding loop
        """
        self._write_gcs_file_lists()
        self._download_images_to_embed()
        self._embed_images()
        self._upload_image_embeddings()
