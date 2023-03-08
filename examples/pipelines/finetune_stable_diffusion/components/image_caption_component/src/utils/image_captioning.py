"""Image captioning"""
import os
import logging
from typing import Iterable, List, Tuple

from tqdm import tqdm
from PIL import Image
from pyarrow.dataset import Scanner

# pylint: disable=import-error
from helpers import storage_helpers, parquet_helpers
from helpers.logger import get_logger
from utils.blip_model import BlipModel

LOGGER = get_logger(name=__name__, level=logging.INFO)

# Set to allow pillow to process images of different sizes
Image.MAX_IMAGE_PIXELS = None


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class KfpPipelineImageCaptioner:
    """kfp image captioner """

    # pylint: disable=too-many-arguments
    def __init__(self, blip_model: BlipModel, parquet_dataset: Scanner, tmp_img_path: str,
                 tmp_dir: str, namespace: str):
        """
        Class that structures the kfp images conversion loop
        Args:
            blip_model (BlipModel): the blip model
            parquet_dataset (Scanner): the scanned parquet dataset
            tmp_img_path (str) the temporary path used to store the downloaded images
            tmp_dir (str): temporary directory where to store
            namespace (str): the dataset namespace of the images to caption
        """
        self.blip_model = blip_model
        self.parquet_dataset = parquet_dataset
        self.tmp_img_path = tmp_img_path
        self.caption_parquet_path = os.path.join(tmp_dir, f'captions_{namespace}.parquet')
        self.download_list_file_path = os.path.join(tmp_dir, f'download_gcs_{namespace}.txt')

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

    def _download_images_to_caption(self):
        """Function that download the images to caption locally"""
        LOGGER.info("Downloading images to caption")
        storage_helpers.copy_files_bulk(self.download_list_file_path,
                                        self.tmp_img_path)
        LOGGER.info("The images to be captioned were successfully downloaded to %s",
                    self.tmp_img_path)

    def _write_parquet_captions(self, min_length: int, max_length: int, beams: int):
        """
        Function that captions the images with BLIP and ranked them with CLIP
        Args:
            min_length (int): the minimum caption length
            max_length (int): the maximum caption length
            beams (int): parameters to increase the caption per image quality but required more
            compute time
        """

        def _img_caption_loop() -> Iterable[Tuple[str, str, List[str]]]:

            LOGGER.info("Starting image captioning")
            for batch in self.parquet_dataset.to_batches():
                # pyarrow's batch_size approximates the number of batches to return
                # Batches may be smaller if there aren’t enough rows in the file and can return zero
                # in some occasions
                if len(batch) > 0:
                    file_uris, file_ids, local_paths = [], [], []
                    for row in tqdm(batch.to_pylist()):
                        file_uri, file_id = row['file_uri'], row['file_id']
                        file_name = os.path.basename(file_uri)
                        local_file_path = os.path.join(self.tmp_img_path, file_name)
                        file_ids.append(file_id)
                        file_uris.append(file_uri)
                        local_paths.append(local_file_path)
                        # caption images
                    captions = self.blip_model.caption_images(
                        image_paths=local_paths,
                        min_length=min_length,
                        max_length=max_length,
                        beams=beams)
                    for file_id, file_uri, caption in zip(file_ids, file_uris, captions):
                        # TODO: decide whether the caption field in caption parquet should be a str
                        #  or a list (many caption per image)
                        yield file_id, file_uri, [caption]

        parquet_helpers.write_captions_parquet(
            caption_parquet_path=self.caption_parquet_path,
            data_iterable_producer=_img_caption_loop
        )

        LOGGER.info("Updated dataset parquet file written to %s", self.caption_parquet_path)
        LOGGER.info("Image conversion completed")

    def start(self, min_length: int, max_length: int, beams: int):
        """
        Function that starts the image captioning loop
        Args:
            min_length (int): the minimum caption length
            max_length (int): the maximum caption length
            beams (int): parameters to increase the caption per image quality but required more
            compute time
        """
        self._write_gcs_file_lists()
        self._download_images_to_caption()
        self._write_parquet_captions(min_length=min_length, max_length=max_length, beams=beams)

    def get_caption_path(self) -> str:
        """
        Function that returns the image caption path
        Returns:
            str: the image caption path
        """
        return self.caption_parquet_path
