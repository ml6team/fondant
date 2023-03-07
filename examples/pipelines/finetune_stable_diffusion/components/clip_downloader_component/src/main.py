"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""

import os
import tempfile
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

from google.cloud import storage
from tqdm import tqdm

# pylint: disable=import-error
from utils.image_downloader import ImageDownloaderInput, ImageDownloader
from helpers.logger import get_logger
from helpers import storage_helpers, parquet_helpers
from helpers.manifest_helpers import DataManifest


def parse_args():
    """Parse component arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',
                        type=str,
                        required=True,
                        help='The run id of the pipeline')
    parser.add_argument('--artifact-bucket',
                        type=str,
                        required=True,
                        help='The GCS bucket used to store the artifacts')
    parser.add_argument('--component-name',
                        type=str,
                        required=True,
                        help='The name of the component')
    parser.add_argument('--project-id',
                        type=str,
                        required=True,
                        help='The id of the gcp-project')
    parser.add_argument('--image-resize',
                        type=str,
                        required=True,
                        help='the size to resize the image')
    parser.add_argument('--timeout',
                        type=int,
                        required=True,
                        help='maximum time (in seconds to wait) when trying to download an image')
    parser.add_argument('--min-image-size',
                        type=int,
                        required=True,
                        help='Minimum size of the image to download (default 0)')
    parser.add_argument('--max-image-area',
                        type=int,
                        required=True,
                        help='maximum area of the image to download (default inf)')
    parser.add_argument('--data-manifest-path',
                        type=str,
                        required=True,
                        help='The previous component manifest path')
    parser.add_argument('--parquet-path-clip-knn-retrieval',
                        type=str,
                        required=True,
                        help='The path to the parquet file containing the urls from knn retrieval')
    parser.add_argument('--parquet-path-clip-centroid-retrieval',
                        type=str,
                        required=True,
                        help='The path to the parquet file containing the urls from centroid '
                             'retrieval')
    parser.add_argument('--data-manifest-path-clip-downloader-component',
                        type=str,
                        required=True,
                        help='Path to the local file containing the gcs path where the output'
                             ' has been stored')

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
def clip_downloader_component(run_id: str,
                              artifact_bucket: str,
                              component_name: str,
                              project_id: str,
                              image_resize: int,
                              timeout: int,
                              min_image_size: int,
                              max_image_area: int,
                              data_manifest_path: str,
                              parquet_path_clip_knn_retrieval: str,
                              parquet_path_clip_centroid_retrieval: str,
                              data_manifest_path_clip_downloader_component: str) -> None:
    """
    Args:
        run_id (str): The run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifact
        component_name (str): the name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        image_resize (int): the size to resize the image
        timeout (int): maximum time (in seconds to wait) when trying to download an image
        min_image_size (int): minimum size of the image to download (default 0)
        max_image_area (int):  maximum area of the image to download (default inf)
        data_manifest_path (str): The previous component manifest path
        parquet_path_clip_centroid_retrieval (str): The path to the parquet file containing the urls
        from centroid retrieval
        parquet_path_clip_knn_retrieval (str): The path to the parquet file containing the urls from
         knn retrieval
        data_manifest_path_clip_downloader_component (str): the current component manifest path
    """
    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info('Started job...')

    # Initialize storage client
    storage_client = storage.Client(project=project_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info('created temporary directory %s', tmp_dir)

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition('-')[0]
        artifact_bucket_blob_path = f"custom_artifact/{component_artifact_dir}/{component_name}"
        logger.info("custom artifact will be uploaded to %s",
                    f'gs://{artifact_bucket}/{artifact_bucket_blob_path}')

        # Initialize local directory
        tmp_image_dir_path = os.path.join(tmp_dir, 'img_dir')
        knn_dataset_parquet_path = os.path.join(tmp_dir, 'knn_dataset.parquet')
        centroid_dataset_parquet_path = os.path.join(tmp_dir, 'centroid.parquet')
        knn_image_dir_path = os.path.join(tmp_dir, 'knn_images')
        centroid_image_dir_path = os.path.join(tmp_image_dir_path, 'centroid_images')
        os.makedirs(knn_image_dir_path, exist_ok=True)
        os.makedirs(centroid_image_dir_path, exist_ok=True)

        # Initialize gcs paths
        downloaded_image_blob = os.path.join(artifact_bucket_blob_path, 'clip_retrieval_images')
        knn_parquet_blob = os.path.join(artifact_bucket_blob_path, 'knn_dataset.parquet')
        centroid_parquet_blob = os.path.join(artifact_bucket_blob_path, 'centroid_dataset.parquet')
        index_parquet_blob = os.path.join(artifact_bucket_blob_path, 'index.parquet')

        # Load manifest
        with open(data_manifest_path) as f:
            manifest_load = json.load(f)
        data_manifest = DataManifest.from_dict(manifest_load)

        # Load clip retrieval parquet files
        with open(parquet_path_clip_knn_retrieval) as f:
            parquet_path_clip_knn_retrieval = f.readline()
        with open(parquet_path_clip_centroid_retrieval) as f:
            parquet_path_clip_centroid_retrieval = f.readline()

        # Copy clip retrieval parquet files (containing clip ids and urls) to local directory
        parquet_clip_knn_retrieval_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, parquet_path_clip_knn_retrieval, tmp_dir)
        parquet_clip_centroid_retrieval_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, parquet_path_clip_centroid_retrieval, tmp_dir)
        # Copy index to local directory
        index_parquet_path = storage_helpers.download_file_from_bucket(
            storage_client, data_manifest.index, tmp_dir)

        # Remove duplicate images based on ids. This duplication may have occurred since querying
        # for the centroid and knn was done separately.
        logger.info('Removing duplicate entries from centroid and KNN.')
        parquet_helpers.remove_common_duplicates(
            dataset_to_filter_path=parquet_clip_knn_retrieval_tmp_path,
            reference_dataset_path=parquet_clip_centroid_retrieval_tmp_path,
            duplicate_columns_name='id',
            tmp_path=tmp_dir)
        logger.info('Duplicates removed')

        # Fetch and download images
        knn_downloader_input = \
            ImageDownloaderInput(source_url=parquet_clip_knn_retrieval_tmp_path,
                                 dest_images=knn_image_dir_path,
                                 dest_dataset_parquet=knn_dataset_parquet_path,
                                 namespace='knn')
        centroid_downloader_input = \
            ImageDownloaderInput(source_url=parquet_clip_centroid_retrieval_tmp_path,
                                 dest_images=centroid_image_dir_path,
                                 dest_dataset_parquet=centroid_dataset_parquet_path,
                                 namespace='centroid')

        img_downloader = ImageDownloader(timeout=timeout,
                                         image_resize=image_resize,
                                         min_image_size=min_image_size,
                                         max_image_area=max_image_area)

        for img_downloader_tuple in tqdm([knn_downloader_input, centroid_downloader_input]):
            logger.info("Stating downloader job for %s dataset", img_downloader_tuple.namespace)
            img_downloader.run(image_downloader_tuple=img_downloader_tuple,
                               tmp_dir=tmp_dir,
                               dest_gcs_uri=f"gs://{artifact_bucket}/{downloaded_image_blob}",
                               index_path=index_parquet_path)
            logger.info('Image download job for %s dataset is complete',
                        img_downloader_tuple.namespace)
        # Print job stat dict that shows how many download job succeeded/failed
        # TODO: integrate output of job stat dict with the dataset metadata in the manifest
        logger.info('job stat dict: %s', img_downloader.get_job_stat_dict())

        # Upload parquet files
        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=index_parquet_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=index_parquet_blob)
        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=centroid_dataset_parquet_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=centroid_parquet_blob)
        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=knn_dataset_parquet_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=knn_parquet_blob)

        # Update manifest
        logger.info('Updating data manifest job')
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.creation_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        data_manifest.metadata.run_id = run_id
        data_manifest.index = f"gs://{artifact_bucket}/{index_parquet_blob}"
        data_manifest.associated_data.dataset['centroid'] = \
            f"gs://{artifact_bucket}/{centroid_parquet_blob}"
        data_manifest.associated_data.dataset['knn'] = \
            f"gs://{artifact_bucket}/{knn_parquet_blob}"
        logger.info('Manifest file updated')

        Path(data_manifest_path_clip_downloader_component).parent.mkdir(parents=True, exist_ok=True)
        Path(data_manifest_path_clip_downloader_component).write_text(data_manifest.to_json())
        logger.info('Manifest file written to %s', data_manifest_path_clip_downloader_component)
        logger.info('Job completed.')


if __name__ == '__main__':
    args = parse_args()
    clip_downloader_component \
        (run_id=args.run_id,
         artifact_bucket=args.artifact_bucket,
         component_name=args.component_name,
         project_id=args.project_id,
         image_resize=args.image_resize,
         timeout=args.timeout,
         min_image_size=args.min_image_size,
         max_image_area=args.max_image_area,
         data_manifest_path=args.data_manifest_path,
         parquet_path_clip_knn_retrieval=
         args.parquet_path_clip_knn_retrieval,
         parquet_path_clip_centroid_retrieval=args.parquet_path_clip_centroid_retrieval,
         data_manifest_path_clip_downloader_component=
         args.data_manifest_path_clip_downloader_component)
