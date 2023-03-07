"""
Common huggingface utils
"""
import logging
import os

# pylint: disable=import-error
from pyarrow.dataset import Scanner
import jsonlines

from helpers import io_helpers, storage_helpers
from helpers.logger import get_logger

logger = get_logger(name=__name__, level=logging.INFO)


# pylint: disable=too-few-public-methods
class SDDatasetCreator:
    """
    Class that creates the dataset for SD
    """

    # pylint: disable=too-many-arguments
    def __init__(self, tmp_path: str, img_path: str):
        """
        Initialize parameters and dataset
        Args:
            tmp_path (str): path where to store temporary files
            img_path (str): the path where to download the images to finetune
        """
        self.tmp_path = tmp_path
        self.img_path = img_path
        self.metadata_file_path = os.path.join(self.img_path, 'metadata.jsonl')

    def _download_images_to_finetune(self, dataset_scanner: Scanner,
                                     download_list_file_path: str) -> None:
        """
        Function that writes downloads the images to finetune on locally
        Args:
            dataset_scanner (Scanner): a scanner of the dataset
            download_list_file_path (str): the text files containing the list of uri to download
        """

        with open(download_list_file_path, "w") as download_list_file:
            for batch in dataset_scanner.to_batches():
                files_uri_list = batch.to_pydict()['file_uri']
                download_list_file.write('\n'.join(files_uri_list))

        storage_helpers.copy_files_bulk(download_list_file_path, self.img_path)
        logger.info("Images to downloaded were downloaded to %s", self.img_path)

    def _write_hf_metadata(self, captions_scanner: Scanner) -> None:
        """
        Function that writes the huggingface metadata file
        Reference:
        https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata:
        ~:text=in%20load_dataset.-,Image%20captioning,-Image%20captioning%20datasets
        Args:
            captions_scanner (Scanner): the captions scanner
        """
        # We join the captions at this stage since it might create duplicates if there are
        # multiple captions per image, we do not want to write them to the text containing the
        # gcs images to download (avoid downloading the same image multiple times)

        with jsonlines.open(self.metadata_file_path, 'a') as writer:
            for table_chunk in captions_scanner.to_batches():
                table_dict = table_chunk.to_pydict()
                # Return file name
                table_dict['file_uri'] = \
                    [io_helpers.get_file_name(file_uri, return_extension=True) for file_uri in
                     table_dict['file_uri']]
                # change format to match the expected value of HF metadata dataset
                # {"file_name":<file_path>, "text": <file_caption>}
                table_dict['file_name'] = table_dict.pop('file_uri')
                table_dict['text'] = table_dict.pop('file_captions')
                # Filter dict to keep only field of interest to create the dataset metadata file
                table_dict = {key: table_dict[key] for key in ['file_name', 'text']}
                # Change dict of lists to list of dicts to write them to json new lines in batch
                table_dict = [dict(zip(table_dict, t)) for t in zip(*table_dict.values())]
                # Unpack caption list to have separate rows per caption (required HF format to have
                # image/caption pair)
                table_dict = [{"file_name": row['file_name'], "text": caption} for row in table_dict
                              for caption in row['text']]
                # pylint: disable=no-member
                writer.write_all(table_dict)

        logger.info("Huggingface metadata file written to %s", self.metadata_file_path)

    def write_dataset(self, dataset_scanner: Scanner, captions_scanner: Scanner, namespace: str):
        """Create the dataset by downloading the images and creating the huggingface dataset
        metadata file
        Args:
            dataset_scanner (Scanner): the filtered dataset scanner
            captions_scanner (Scanner): the filtered captions scanner
            namespace (str): the dataset namespace
        """
        download_list_file_path = os.path.join(self.tmp_path, f"download_list_gcs_{namespace}.txt")
        self._download_images_to_finetune(dataset_scanner, download_list_file_path)
        self._write_hf_metadata(captions_scanner)
