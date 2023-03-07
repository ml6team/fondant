"""Image classification"""
import os
import time
import logging
from typing import List
from itertools import compress

# pylint: disable=import-error
import torch
import numpy as np
import pyarrow.compute as pc
from pyarrow.dataset import Scanner
from tqdm import tqdm
from PIL import Image

import clip
from helpers import storage_helpers, parquet_helpers, io_helpers
from helpers.logger import get_logger
from classifiers.single_component import single_component_ensemble
from classifiers.clean_cut import CCFClassifierInference

logger = get_logger(name=__name__, level=logging.INFO)

# Set to allow pillow to process images of different sizes
Image.MAX_IMAGE_PIXELS = None


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class KfpPipelineImageClassification:
    """kfp image classification """

    # pylint: disable=too-many-arguments
    def __init__(self, ccf_classifier_path: str, clip_model_path: str, tmp_path: str):
        """
        Initialize kfp classifier class
        Args:
            ccf_classifier_path (str): the path to the clean cut model containing weights and
                configs
            clip_model_path (str): the path to the clip model
            tmp_path (str): the path to store temporary files
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info('Initializing models with %s device', self.device)

        self.clip_model_vit, self.clip_preprocess_vit = clip.load(clip_model_path,
                                                                  device=self.device)
        logger.info('Clip model initialized')
        self.ccf_classifier_path = ccf_classifier_path
        self.ccf_model_name = os.path.basename(self.ccf_classifier_path)

        self.clean_cut_classifier = CCFClassifierInference(
            weights=os.path.join(self.ccf_classifier_path, f'{self.ccf_model_name}.pt'),
            config=os.path.join(self.ccf_classifier_path, f'{self.ccf_model_name}-config.json'),
            device=self.device)
        logger.info('Clean cut model initialized')
        self.tmp_path = tmp_path

    @staticmethod
    def _write_gcs_file_lists(parquet_dataset: Scanner, download_list_file_path: str):
        """
        Function that writes the gcs download and upload list files for bulk download/upload
        Args:
            download_list_file_path (str): path to text file with url of paths to download
            parquet_dataset (Scanner): the scanned parquet dataset
        """
        with open(download_list_file_path, "w") as download_list_file:
            for batch in parquet_dataset.to_batches():
                for row in batch.to_pylist():
                    file_uri = row['file_uri']
                    download_list_file.write(file_uri + "\n")

        logger.info("GCS download file list written to %s", download_list_file_path)

    @staticmethod
    def _download_images_to_classify(download_list_file_path: str, tmp_img_path: str):
        """
        Function that download the images to caption locally
        Args:
            download_list_file_path (str): path to text file with url of paths to download
            tmp_img_path (str): path where to store the temporary images
        """
        logger.info("Downloading images to caption")
        storage_helpers.copy_files_bulk(download_list_file_path,
                                        tmp_img_path)
        logger.info("The images to be classified were successfully downloaded to %s", tmp_img_path)

    # pylint: disable=too-many-locals
    def _clean_cut_classification(self, parquet_dataset: Scanner, tmp_img_path: str) -> List[str]:
        """
        Function that classifies the images with the clean cut classifier
        Args:
            parquet_dataset (Scanner): the filtered parquet dataset
            tmp_img_path (str): path where to store the temporary images
        Returns:
            List[str]: the filtered indices
        """
        logger.info("Starting image classification with clean cut classifier.")
        start = time.time()
        filtered_index = []
        for batch in parquet_dataset.to_batches():
            # pyarrow's batch_size approximates the number of batches to return
            # Batches may be smaller if there aren’t enough rows in the file and can return zero
            # in some occasions
            if len(batch) > 0:
                img_list, file_ids, local_paths = [], [], []
                for row in tqdm(batch.to_pylist()):
                    file_uri, file_id = row['file_uri'], row['file_id']
                    file_name = io_helpers.get_file_name(file_uri, return_extension=True)
                    local_file_path = os.path.join(tmp_img_path, file_name)
                    img_list.append(self.clip_preprocess_vit(Image.open(local_file_path)))
                    local_paths.append(local_file_path)
                    file_ids.append(file_id)
                # pylint: disable=no-member
                img_stack = torch.tensor(np.stack(img_list), device=self.device)
                embeddings = self.clip_model_vit.encode_image(img_stack).cpu().detach().numpy()

                clean_cut_bool_arr = self.clean_cut_classifier.predict(embeddings)

                # remove element from index that are classified as not clean cut
                filtered_idx_batch = list(compress(file_ids, clean_cut_bool_arr.tolist()))
                filtered_index.extend(filtered_idx_batch)

        end = int((time.time() - start) / 60)
        logger.info("Clean cut classification complete in %s minutes", end)
        return filtered_index

    @staticmethod
    def _single_component_classification(parquet_dataset: Scanner, tmp_img_path: str) \
            -> List[str]:
        """
        Function that classifies the images with the connected component classifier
        Args:
            parquet_dataset (Scanner): the scanned parquet dataset
            tmp_img_path (str): path where to store the temporary images
        Returns:
            List[str]: the filtered indices
        """
        logger.info("Starting image classification with single component classifier.")
        start = time.time()
        filtered_index = []

        for batch in parquet_dataset.to_batches():
            # pyarrow's batch_size approximates the number of batches to return
            # Batches may be smaller if there aren’t enough rows in the file and can return zero
            # in some occasions
            if len(batch) > 0:
                for row in tqdm(batch.to_pylist()):
                    file_uri, file_id = row['file_uri'], row['file_id']
                    file_name = io_helpers.get_file_name(file_uri, return_extension=True)
                    local_file_path = os.path.join(tmp_img_path, file_name)
                    if single_component_ensemble(local_file_path):
                        filtered_index.append(file_id)

        end = int((time.time() - start) / 60)
        logger.info("Single component classification complete in %s minutes", end)
        return filtered_index

    def start(self, index: List[str], parquet_dataset_path: str, namespace: str,
              batch_size_clean_cut: int) -> List[str]:
        """
        Function that starts the image classification loop
        Args:
            namespace (str): the namespace of the images to classify
            index (List[str]): the dataset index
            parquet_dataset_path (str): the path to the parquet dataset
            batch_size_clean_cut (int): the batch size to pass to the clean cut classifier
        Returns:
            List[str]: the filtered indices
        """

        # Initialize temporary files
        tmp_img_path = os.path.join(self.tmp_path, f"tmp_images_{namespace}")
        download_list_file_path = os.path.join(tmp_img_path, f"img_download_{namespace}.txt")
        os.makedirs(tmp_img_path, exist_ok=True)

        # Filter initial dataset based on index
        filters = (pc.field("file_id").isin(index))
        parquet_dataset_generator = parquet_helpers.filter_parquet_file(
            file_path=parquet_dataset_path,
            filters=filters,
            batch_size=batch_size_clean_cut)
        start_nb_images = parquet_dataset_generator.count_rows()
        logger.info("A total of %s images to classify", start_nb_images)

        # Download images to classify
        self._write_gcs_file_lists(parquet_dataset_generator, download_list_file_path)
        self._download_images_to_classify(download_list_file_path, tmp_img_path)

        # clean cut classifier
        index = self._clean_cut_classification(parquet_dataset_generator, tmp_img_path)
        nb_images_filtered_clean_cut = start_nb_images - len(index)
        logger.info("(1/2) %s out of %s images were filtered with the clean cut classifier",
                    nb_images_filtered_clean_cut, start_nb_images)

        # connected component classifier
        filters = (pc.field("file_id").isin(index))
        parquet_dataset_generator = parquet_helpers.filter_parquet_file(
            file_path=parquet_dataset_path,
            filters=filters)
        index = self._single_component_classification(parquet_dataset_generator, tmp_img_path)
        nb_images_filtered_connected_component = start_nb_images - len(index)
        logger.info(
            "(2/2) %s out of %s images were filtered with the single component cut classifier",
            nb_images_filtered_connected_component, start_nb_images)

        # global statistics
        nb_images_filtered_total = \
            nb_images_filtered_connected_component + nb_images_filtered_clean_cut
        percentage_filtered = round(100 * (nb_images_filtered_total / start_nb_images), 2)
        logger.info(
            "A total of %s images out of %s (%s%%) were filtered",
            nb_images_filtered_connected_component,
            start_nb_images,
            percentage_filtered)

        return index
