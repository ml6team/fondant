"""
Write and read files from Google Cloud Storage (GCS)
"""
import hashlib
import subprocess  # nosec
import logging
import os
from urllib.parse import urlparse
from typing import Optional, List, Tuple

from google.cloud import storage

# pylint: disable=import-error
from .logger import get_logger

LOGGER = get_logger(name=__name__, level=logging.INFO)


def unique_name_from_str(str_to_encode: str, last_idx: int = 12) -> str:
    """
    Function that generates a unique id name from a given string
    Args:
        str_to_encode (str): the string to encode
        last_idx (int): the number of characters of the unique id
    Returns:
        str: the encoded string
    """
    string = str_to_encode.encode('utf-8')
    unique_name = hashlib.sha384(string).hexdigest()[0:last_idx]
    return unique_name


def decode_gcs_url(gcs_path: str) -> Tuple[str, str]:
    """
    Function that decodes a GCS path to a bucket and blob path
    (i.e. file path)
    Args:
        gcs_path (str): the GCS URL (starting with https://) or URI (starting with gs://)
    Returns:
        Tuple[str, str]: a tuple containing the bucket and blob path
    """
    parsed_url = urlparse(gcs_path)
    if parsed_url.scheme == 'gs':
        bucket, blob_path = parsed_url.hostname, parsed_url.path[1:]
    else:
        path = parsed_url.path[1:].split('/', 1)
        bucket, blob_path = path[0], path[1]
    return bucket, blob_path


def get_blob_list(storage_client: storage.Client, bucket_name: str,
                  prefix: Optional[str] = None) -> List[storage.Blob]:
    """
    Function that returns all the blobs in a bucket.
    Optionally you can pass a  within a given prefix (i.e. folder path) to only list blobs within
     that prefix
    Args:
        storage_client (storage.Client): the gcs storage client
        bucket_name (str): the name of the bucket
        prefix (Optional[str]): the prefix of the bucket. If not specified, only the blobs within
        the given bucket will be listed
    Returns:
        List[storage.Blob]: a list of blob objects
    """

    blobs_iterator = storage_client.list_blobs(bucket_name, prefix=prefix)
    blob_list = [blob for blob in blobs_iterator if blob.size != 0]
    return blob_list


def get_blob_metadata(storage_client: storage.Client, bucket_name: str,
                      prefix: Optional[str] = None, id_prefix: Optional[str] = None) -> tuple:
    """
    Function that returns all the blobs in a bucket.
    Optionally you can pass a  within a given prefix (i.e. folder path) to only list blobs
     within that prefix
    Args:
        storage_client (storage.Client): the gcs storage client
        bucket_name (str): the name of the bucket
        prefix (Optional[str]): the prefix of the bucket. If not specified, only the blobs within
        the given bucket will be listed
        id_prefix (Optional[str]): a prefix to add to the file id
    Returns:
        tuple: tuple containing relevant metadata
    """

    blob_list = get_blob_list(storage_client, bucket_name, prefix)

    for blob in blob_list:
        # id taken from the blob id of format ('<gcs_path>/<id_number>')
        blob_id = unique_name_from_str(str(blob.id))
        if id_prefix:
            file_id = f'{id_prefix}_{blob_id}'
        else:
            file_id = blob_id
        file_size = blob.size
        file_extension = blob.content_type.rsplit('/')[1]
        file_uri = f"gs://{bucket_name}/{blob.name}"
        yield file_uri, file_id, file_size, file_extension


def get_blob_id(storage_client: storage.Client, bucket_name: str,
                prefix: Optional[str] = None, id_prefix: Optional[str] = None) -> str:
    """
    Function that returns blob id from a blob list
    Args:
        storage_client (storage.Client): the gcs storage client
        bucket_name (str): the name of the bucket
        prefix (Optional[str]): the prefix of the bucket. If not specified, only the blobs within
        the given bucket will be listed
        id_prefix (Optional[str]): a prefix to add to the file id
    Returns:
        tuple: tuple containing file id
    """
    blob_list = get_blob_list(storage_client, bucket_name, prefix)

    for blob in blob_list:
        # id taken from the blob id of format ('<gcs_path>/<id_number>')
        blob_id = unique_name_from_str(str(blob.id))
        file_id = f'{id_prefix}_{blob_id}'
        yield file_id


def download_file_from_bucket(storage_client: storage.Client, gcs_uri: str,
                              folder_to_download: str) -> str:
    """
    Function that downloads a file from GCS to a local directory
    Args:
        storage_client (storage.Client): the gcs storage client
        gcs_uri (str): the gcs path to download (URL or URI format)
        folder_to_download (str): the local folder to download the file to
    Returns:
        str: the full path where the file was downloaded
    """
    bucket, blob_path = decode_gcs_url(gcs_uri)
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(blob_path)
    path_to_download = os.path.join(os.path.dirname(folder_to_download),
                                    os.path.basename(blob_path))
    blob.download_to_filename(path_to_download)

    return path_to_download


def copy_files_bulk(filelist_source: str, destination_path: str):
    """
    Copies files from a source path to a destination path sing parallel multi-threading
    for increased efficiency
    Args:
        filelist_source (str): path to file list containing one-per-line list of GCS URIs or a list
         of local paths to be copied
        destination_path (str): Path to where the files should be copied (can be a local path or a
        GCS URI).
    """
    LOGGER.info("Copying files from %s to %s ", filelist_source, destination_path)
    pipe_file_list = subprocess.Popen(["cat", filelist_source], stdout=subprocess.PIPE)  # nosec
    subprocess.call(  # nosec
        ['gsutil', '-o', '"GSUtil:use_gcloud_storage=True"', '-q', '-m', 'cp', '-I',
         destination_path], stdin=pipe_file_list.stdout)
    LOGGER.info("Copying files from %s to %s [DONE]", filelist_source, destination_path)


def copy_folder_bulk(source_path: str, destination_path: str):
    """Copies a folder from a source path to a destination path sing parallel multi-threading
    for increased efficiency
    Args:
        source_path (str): Path from where the files should be copied (can be a local path or a
         GCS URI).
        destination_path (str): Path to where the files should be copied (can be a local path or a
        GCS URI).
    """
    LOGGER.info("Copying folder from %s to %s ", source_path, destination_path)
    subprocess.run(  # nosec
        ["gsutil", '-o', '"GSUtil:use_gcloud_storage=True"', '-q', "-m", "cp", "-r", source_path,
         destination_path],
        check=True)
    LOGGER.info("Copying folder from %s to %s [DONE]", source_path, destination_path)


def upload_file_to_bucket(storage_client: storage.Client, file_to_upload_path: str,
                          bucket_name: str, blob_path: str) -> None:
    """
    Upload file to bucket
    Args:
        storage_client (storage.Client): the gcs storage client
        file_to_upload_path (str): path of file to upload to the bucket
        bucket_name (str): name of the bucket to upload the data to
        blob_path (str): the path to the blob to upload the data within the specified bucket
    """

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file_to_upload_path)

    LOGGER.info(" %s uploaded to GCS at gs://%s/%s'.",
                file_to_upload_path, bucket_name, blob_path)


def copy_blob(storage_client: storage.Client,
              bucket_name: str, blob_path: str, destination_bucket_name: str,
              destination_blob_name: str):
    """
    Function that copies a blob from one bucket to another
    Args:
        storage_client (storage.Client): the gcs storage client
        bucket_name (str): the source bucket name
        blob_path (str): the source blob name
        destination_bucket_name (str): the destination bucket name
        destination_blob_name (str): the destination blob name
    """
    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_path)
    destination_bucket = storage_client.bucket(destination_bucket_name)
    source_bucket.copy_blob(source_blob, destination_bucket, destination_blob_name)
