"""
Image conversion utils
"""

import io
import os
import logging
from typing import Union

import cairosvg
import pyarrow.dataset as ds
from tqdm import tqdm
from PIL import Image as img
from PIL.Image import Image

# # pylint: disable=import-error
from helpers import storage_helpers, parquet_helpers, io_helpers
from helpers.logger import get_logger

# Set to allow pillow to process images of different sizes
img.MAX_IMAGE_PIXELS = None

LOGGER = get_logger(name=__name__, level=logging.INFO)


def convert_png_to_jpeg(image_source: Union[str, Image]) -> Image:
    """
    Function that converts images from png to jpg
    Args:
        image_source (Union[str,Image]): png image source or Pillow Image object
    Returns:
        Image: Image in jpeg format
    """

    if isinstance(image_source, str):
        image_to_convert = img.open(image_source).convert("RGBA")
    else:
        image_to_convert = image_source
    converted_image = img.new("RGB", image_to_convert.size, (255, 255, 255))
    converted_image.paste(image_to_convert, mask=image_to_convert)

    return converted_image


def convert_svg_to_jpeg(img_path: str, output_width: int, output_height: int) -> Image:
    """
    Function that converts images from svg to jpg
    Args:
        img_path (str): the folder path where the source images are located
        output_width (int): the desired output width of the svg image
        output_height (int): the desired output height of the svg image
    Returns:
        Image: Image in jpeg format
    """
    # SVG has to be converted to png first before converting to jpg
    png_bytes = cairosvg.svg2png(url=img_path, output_width=output_width,
                                 output_height=output_height)
    converted_image = img.open(io.BytesIO(png_bytes))
    converted_image = convert_png_to_jpeg(converted_image)

    return converted_image


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class KfpPipelineImageConverter:
    """kfp images conversion """

    # pylint: disable=too-many-arguments
    def __init__(self, parquet_dataset: ds.dataset, updated_parquet_dataset_path: str,
                 destination_gcs_uri: str, tmp_img_path: str, file_extensions: list,
                 index_ids: list, download_list_file_path: str,
                 upload_list_file_path: str):
        """
        Class that structures the kfp images conversion loop
        Args:
            parquet_dataset (pq.ParquetDataset): the parquet dataset containing the list of images
             to convert
            updated_parquet_dataset_path (str): the path to write the updated parquet path after
              conversion
            destination_gcs_uri (str): the gcs uri to copy the files to.
             It can indicate a bucket (e.g. gs://my-bucket) or a bucket with a blob path
             (e.g gs://my-bucket/my-blob-path/)
            tmp_img_path (str) the temporary path used to store the downloaded and converted images
            file_extensions (list): a list containing the image extensions that need to be converted
            index_ids (list): list of index ids that contain the images of interest
            download_list_file_path (str): path to file list containing one-per-line list of GCS
             URIs to download
            upload_list_file_path (str): path to file list containing one-per-line list of local
             file paths to upload
        """
        self.parquet_dataset = parquet_dataset
        self.updated_parquet_dataset_path = updated_parquet_dataset_path
        self.destination_gcs_uri = destination_gcs_uri
        self.tmp_img_path = tmp_img_path
        self.file_extensions = file_extensions
        self.index_ids = index_ids
        self.download_list_file_path = download_list_file_path
        self.upload_list_file_path = upload_list_file_path

    def _write_gcs_file_lists(self):
        """
        Function that writes the gcs download and upload list files for bulk download/upload
        """
        with open(self.download_list_file_path, "w") as download_list_file, \
                open(self.upload_list_file_path, "w") as upload_list_file:
            for batch in self.parquet_dataset.to_batches():
                for row in batch.to_pylist():
                    file_extension, file_id, file_uri = row['file_extension'], row['file_id'], row[
                        'file_uri']
                    if file_id in self.index_ids and file_extension in self.file_extensions:
                        converted_img_name = f"{io_helpers.get_file_name(file_uri)}.jpg"
                        local_file_path = os.path.join(self.tmp_img_path, converted_img_name)
                        download_list_file.write(file_uri + "\n")
                        upload_list_file.write(local_file_path + "\n")

        LOGGER.info("GCS download file list written to %s", self.download_list_file_path)
        LOGGER.info("GCS upload file list written to %s", self.upload_list_file_path)

    def _download_imgs_to_convert(self):
        """Function that download the images to be converted locally"""
        LOGGER.info("Downloading images to convert")
        storage_helpers.copy_files_bulk(self.download_list_file_path,
                                        self.tmp_img_path)
        LOGGER.info("The images to be converted were successfully downloaded to %s",
                    self.tmp_img_path)

    def _upload_converted_imgs(self):
        """Function that uploads the images to gcs """

        LOGGER.info("Uploading converted images")
        storage_helpers.copy_files_bulk(self.upload_list_file_path,
                                        self.destination_gcs_uri)
        LOGGER.info("The converted images were uploaded to %s", self.destination_gcs_uri)

    def _convert_images_to_jpeg(self, svg_image_width: int, svg_image_height: int):
        """
        Function that converts the images to jpeg format and saves them locally. The updated
        metadata is then written to new updated the parquet dataset file.
        Args:
            svg_image_width (int): the desired width to scale the converted SVG image to
            svg_image_height (int): the desired height to scale the converted SVG image to
        """

        LOGGER.info("Image conversion started")

        def _img_conversion_loop(**kwargs):
            """Function that loops through the images and converts them if they are located
            in the index ids and if they match the specified image formats that need to be
            converted"""
            for batch in tqdm(self.parquet_dataset.to_batches()):
                for row in tqdm(batch.to_pylist()):
                    file_extension, file_id, file_uri, file_size = \
                        row['file_extension'], row['file_id'], row['file_uri'], row['file_size']

                    if file_id in self.index_ids and file_extension in self.file_extensions:
                        img_name = os.path.basename(file_uri)
                        local_file_path = os.path.join(self.tmp_img_path, img_name)
                        # convert image
                        if file_extension.lower() == 'png':
                            image = convert_png_to_jpeg(local_file_path)
                        elif file_extension.lower() == 'svg':
                            image = convert_svg_to_jpeg(local_file_path, kwargs['svg_image_width'],
                                                        kwargs['svg_image_height'])
                        else:
                            raise NotImplementedError(
                                f"Conversion to jpg for format {file_extension} is"
                                f"not implemented. Available conversions are"
                                f" 'png_to_jpeg' and 'svg_to_jpeg'")

                        # Store img_temp_save_path image locally
                        file_extension = "jpg"
                        img_name = f"{io_helpers.get_file_name(file_uri)}.{file_extension}"
                        img_temp_save_path = os.path.join(self.tmp_img_path, img_name)
                        image.save(img_temp_save_path)
                        file_size = os.path.getsize(img_temp_save_path)
                        file_uri = f"{self.destination_gcs_uri}/{img_name}"

                    yield file_uri, file_id, file_size, file_extension

        parquet_helpers.write_dataset_parquet(
            dataset_parquet_path=self.updated_parquet_dataset_path,
            data_iterable_producer=_img_conversion_loop,
            svg_image_height=svg_image_height,
            svg_image_width=svg_image_width)

        LOGGER.info("Updated dataset parquet file written to %s",
                    self.updated_parquet_dataset_path)
        LOGGER.info("Image conversion completed")

    def start(self, svg_image_width: int, svg_image_height: int):
        """
        Function that executes the kfp image conversion steps
        Args:
            svg_image_width (int): the desired width to scale the converted SVG image to
            svg_image_height (int): the desired height to scale the converted SVG image to
        """
        self._write_gcs_file_lists()
        self._download_imgs_to_convert()
        self._convert_images_to_jpeg(svg_image_width, svg_image_height)
        self._upload_converted_imgs()
