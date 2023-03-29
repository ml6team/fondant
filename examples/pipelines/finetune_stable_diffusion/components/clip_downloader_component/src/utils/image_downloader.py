"""Image downloader utils"""
import subprocess  # nosec
import os
import json
import logging
from collections import namedtuple
from typing import Dict

# pylint: disable=import-error
from helpers.logger import get_logger
from helpers import parquet_helpers, storage_helpers, io_helpers

logger = get_logger(name=__name__, level=logging.INFO)

ImageDownloaderInput = namedtuple(
    'ImageDownloaderInput',
    ['source_url', 'dest_images', 'dest_dataset_parquet', 'namespace'])


class ImageDownloader:
    """
    Class for the image downloader that downloads the images from a parquet url files. The class
    also contains functionalities for writing new parquet and indicies files and uploading the
    downloaded files to GCS
    """

    def __init__(self, timeout: int, image_resize: int, min_image_size: int, max_image_area: int):
        """
        Image downloader
        Args:
            timeout (int): maximum time (in seconds to wait) when trying to download an image
            image_resize (int): the size to resize the image
            min_image_size (int): minimum size of the image to download
             (considers the min of width and height)
            max_image_area (int): The maximum area (nr of pixels) of the images to download

        """
        self.timeout = timeout
        self.image_resize = image_resize
        self.min_image_size = min_image_size
        self.max_image_area = max_image_area
        # status dict field contains detailed error logs of failed download images and their counts
        self.global_stat_dict = {
            "count": 0,
            "successes": 0,
            "failed_to_download": 0,
            "failed_to_resize": 0,
            "status_dict": {},
        }

    def _start_download(self, src_parquet_file: str, dest_img_tmp_path: str):
        """
        Function that starts downloading the images
        Args:
            src_parquet_file (str): the source parquet files where the image urls are located
            dest_img_tmp_path (str):  the root folder where the images will be downloaded
            (will contain different image shards)
        """
        logger.info('Starting image download job')
        subprocess.call(['img2dataset',  # nosec
                         '--url_list', src_parquet_file,
                         '--output_folder', dest_img_tmp_path,
                         '--thread_count', '128',
                         '--processes_count', '8',
                         '--output_format', 'files',
                         '--encode_format', 'jpg',
                         '--timeout', str(self.timeout),
                         '--image_size', str(self.image_resize),
                         '--resize_mode', 'border',
                         '--input_format', 'parquet',
                         '--min_image_size', str(self.min_image_size),
                         '--max_image_area', str(self.max_image_area),
                         '--url_col', 'url',
                         '--distributor', 'multiprocessing',
                         '--number_sample_per_shard', '1000'])
        logger.info('Download job complete')

    def _update_download_job_stats(self, dest_img_tmp_path: str):
        """
        Function that updates that download job statics path. 'img2dataset` creates multiple json
        file with stats for each download partition it creates
        Args:
            dest_img_tmp_path (str): the root folder where the images will be downloaded
            (will contain different image shards)
        """
        logger.info('Updating global stat dict')

        def _update_status_dict():
            for status_key, status_value in stat_dict['status_dict'].items():
                if status_key in self.global_stat_dict['status_dict']:
                    self.global_stat_dict['status_dict'][status_key] += status_value
                else:
                    self.global_stat_dict['status_dict'][status_key] = status_value

        # multiple stat files are created for each shard
        stats_files = [file for file in os.listdir(dest_img_tmp_path) if "stats.json" in file]
        for stat_file in stats_files:
            with open(os.path.join(dest_img_tmp_path, stat_file)) as file:
                stat_dict = json.load(file)
                for key in self.global_stat_dict:
                    # Add detailed error logs
                    if key == 'status_dict' and 'status_dict' in stat_dict:
                        _update_status_dict()
                    # add global status logs
                    else:
                        self.global_stat_dict[key] += stat_dict[key]

        logger.info('Update complete')

    @staticmethod
    def _rename_downloaded_images(dest_img_tmp_path: str, namespace: str, index_path: str,
                                  tmp_dir: str):
        """
        Function that renames the downloaded images by appending a namespace to them. The new image
        ids are appended to the index parquet file.
        Args:
            dest_img_tmp_path (str): the root folder where the images will be downloaded
            (will contain different image shards)
            namespace (str): the namespace prefix to append to the name of uploaded files
            index_path (str): the path to the index parquet file
            tmp_dir (str): the path where to write the temporary parquet files
        """
        logger.info('Renaming downloaded images in directory %s with namespace "%s" as prefix',
                    dest_img_tmp_path, namespace)

        def _img_rename_loop():
            sub_img_folders = [folder for folder in os.listdir(dest_img_tmp_path) if
                               os.path.isdir(os.path.join(dest_img_tmp_path, folder))]
            for sub_img_folder in sub_img_folders:
                sub_img_dir = os.path.join(dest_img_tmp_path, sub_img_folder)
                for file in os.listdir(sub_img_dir):
                    if file.endswith('.jpg'):
                        new_file_name = f"{namespace}_{io_helpers.get_file_name(file)}"
                        src = os.path.join(*[dest_img_tmp_path, sub_img_folder, file])
                        dest = os.path.join(
                            *[dest_img_tmp_path, sub_img_folder, f"{new_file_name}.jpg"])
                        os.rename(src, dest)
                        yield new_file_name

        parquet_helpers.append_to_parquet_file(
            parquet_path=index_path,
            data_iterable=_img_rename_loop(),
            tmp_path=tmp_dir)

        logger.info('Renaming task complete')

    @staticmethod
    def _write_image_metadata(dest_img_tmp_path: str, dest_gcs_uri: str, dataset_path: str,
                              upload_file_txt: str):
        """
        Function that writes the parquet dataset (containing) metadata of newly downloaded images.
        Thos function also writes the url of images into a gcs text file to upload the images.
        Args:
            dest_img_tmp_path (str): the root folder where the images will be downloaded
            (will contain different image shards
            dest_gcs_uri (str): the destination where to write images
            dataset_path (str): the path to the dataset parquet path
            upload_file_txt (str): the path where to write the urls of files to upload
        """
        logger.info('Writing metadata for images in directory %s to %s', dest_img_tmp_path,
                    dataset_path)

        def _write_metadata_loop():
            sub_img_folders = [folder for folder in os.listdir(dest_img_tmp_path) if
                               os.path.isdir(os.path.join(dest_img_tmp_path, folder))]
            with open(upload_file_txt, "w") as upload_file:
                for sub_img_folder in sub_img_folders:
                    sub_img_dir = os.path.join(dest_img_tmp_path, sub_img_folder)
                    for file in os.listdir(sub_img_dir):
                        if file.endswith('.jpg'):
                            file_id = io_helpers.get_file_name(file)
                            local_uri = os.path.join(*[dest_img_tmp_path, sub_img_folder, file])
                            file_size = os.path.getsize(local_uri)
                            file_uri = f"{dest_gcs_uri}/{file}"
                            upload_file.write(local_uri + "\n")
                            yield file_uri, file_id, file_size, "jpg"

        parquet_helpers.write_dataset_parquet(
            dataset_parquet_path=dataset_path,
            data_iterable_producer=_write_metadata_loop)

        logger.info('Metadata written')

    def run(self, image_downloader_tuple: ImageDownloaderInput, tmp_dir: str, dest_gcs_uri: str,
            index_path: str):
        """
        Start and image download loop that downloads the images, uploads them to GCS under
            a new name space and updates the parquet files
        Args:
            image_downloader_tuple (ImageDownloaderInput): tuple containing attributes for the image
            download functionality
            tmp_dir (str): the path where to write the temporary files
            dest_gcs_uri (str): the destination where to write the images
            index_path (str): the path to the index parquet file
        """

        source_url, dest_images, dest_parquet, namespace = image_downloader_tuple.source_url, \
            image_downloader_tuple.dest_images, image_downloader_tuple.dest_dataset_parquet, \
            image_downloader_tuple.namespace
        upload_file_txt = os.path.join(tmp_dir, f"gcs_upload_{namespace}.txt")

        self._start_download(source_url, dest_images)
        self._update_download_job_stats(dest_images)
        self._rename_downloaded_images(dest_img_tmp_path=dest_images,
                                       namespace=namespace,
                                       index_path=index_path,
                                       tmp_dir=tmp_dir)
        self._write_image_metadata(dest_img_tmp_path=dest_images,
                                   dest_gcs_uri=dest_gcs_uri,
                                   dataset_path=dest_parquet,
                                   upload_file_txt=upload_file_txt)
        logger.info('Uploading downloaded images to %s', dest_gcs_uri)
        storage_helpers.copy_files_bulk(upload_file_txt, dest_gcs_uri)
        logger.info('Upload complete')

    def get_job_stat_dict(self) -> Dict[str, int]:
        """
        Function that returns the global job state dict which contains information about the run
        job statistics
        Returns:
            Dict[str, int]: the download global job statistics
        """
        return self.global_stat_dict
