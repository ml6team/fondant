"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""

import os
import tempfile
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm
from google.cloud import storage

# pylint: disable=import-error
from helpers.logger import get_logger
from helpers import storage_helpers, parquet_helpers
from helpers.manifest_helpers import DataManifest
from utils.embedding_utils import get_average_embedding
from utils.knn_service import ClipRetrievalLaion5B


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
    parser.add_argument('--laion-index-url',
                        type=str,
                        required=True,
                        help='contains the indices of the metadata. Those indices need to be '
                             'transformed in case you decide to use only a subset of the dataset')
    parser.add_argument('--laion-metadata-url',
                        type=str,
                        required=True,
                        help='The id of the gcp-project')
    parser.add_argument('--nb-images-knn',
                        type=int,
                        required=True,
                        help='The number of images to return via KNN method')
    parser.add_argument('--nb-images-centroid',
                        type=int,
                        required=True,
                        help='The number of images to return via the centroid method')
    parser.add_argument('--data-manifest-path',
                        type=str,
                        required=True,
                        help='The previous component manifest path')
    parser.add_argument('--data-manifest-path-clip-retrieval-component',
                        type=str,
                        required=True,
                        help='The path to the output manifest file')
    parser.add_argument('--parquet-path-clip-knn-retrieval',
                        type=str,
                        required=True,
                        help='The path to the parquet file containing the urls from knn retrieval')
    parser.add_argument('--parquet-path-clip-centroid-retrieval',
                        type=str,
                        required=True,
                        help='The path to the parquet file containing the urls from centroid '
                             'retrieval')

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
def clip_retrieval_component(run_id: str,
                             artifact_bucket: str,
                             component_name: str,
                             project_id: str,
                             laion_index_url: str,
                             laion_metadata_url: str,
                             nb_images_knn: int,
                             nb_images_centroid: int,
                             data_manifest_path: str,
                             data_manifest_path_clip_retrieval_component: str,
                             parquet_path_clip_knn_retrieval: str,
                             parquet_path_clip_centroid_retrieval: str) -> None:
    """
    Args:
        run_id (str): The run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifact
        component_name (str): the name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        laion_index_url (str):  contains the indices of the metadata. Those indices need to be
         transformed in case you decide to use only a subset of the dataset
        laion_metadata_url (str): url to the metadata of laion dataset metadata (arrow format). It
         can either contain a subset of the laion 5b metadata (e.g. laion-en) or all of the metadata
        nb_images_knn (int): The number of images to return via knn method (per image)
        nb_images_centroid (int): The number of images to return via the centroid method
        data_manifest_path (str): The previous component manifest path
        data_manifest_path_clip_retrieval_component (str): the current component output path
        parquet_path_clip_centroid_retrieval (str): The path to the parquet file containing the urls
         from centroid retrieval
        parquet_path_clip_knn_retrieval (str): The path to the parquet file containing the urls
         from knn retrieval
    """
    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info('Started job...')

    # Initialize storage client
    storage_client = storage.Client(project=project_id)

    # This component uses local SSD mounting to enable faster querying of the laion dataset. The
    # local SSDs are mounted in the cache directory
    laion_index_folder = os.path.join('/cache', 'laion_dataset')
    laion_metadata_folder = os.path.join(laion_index_folder, 'metadata')
    laion_indices_folder = os.path.join(laion_index_folder, 'image.index')
    os.makedirs(laion_metadata_folder, exist_ok=True)
    os.makedirs(laion_indices_folder, exist_ok=True)

    # Download laion indices and metadata from storage
    start_indices = time.time()
    storage_helpers.copy_folder_bulk(laion_index_url, laion_indices_folder)
    logger.info('Laion index download complete: it took %s minutes to download the laion indices',
                round((time.time() - start_indices) / 60))
    start_metadata = time.time()
    storage_helpers.copy_folder_bulk(laion_metadata_url, laion_metadata_folder)
    logger.info(
        'Laion metadata download complete: it took %s minutes to download the laion metadata',
        round((time.time() - start_metadata) / 60))

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info('created temporary directory %s', tmp_dir)

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition('-')[0]
        artifact_bucket_blob_path = f"custom_artifact/{component_artifact_dir}/{component_name}"
        logger.info("custom artifact will be uploaded to %s",
                    f'gs://{artifact_bucket}/{artifact_bucket_blob_path}')

        tmp_embedding_dir_path = os.path.join(tmp_dir, 'embed_dir')
        os.makedirs(tmp_embedding_dir_path, exist_ok=True)
        laion_indices_folder = os.path.join(tmp_dir, 'index_file.json')
        knn_retrieval_path = os.path.join(tmp_dir, 'clip_knn.parquet')
        centroid_retrieval_path = os.path.join(tmp_dir, 'clip_centroid.parquet')
        knn_retrieval_blob_path = os.path.join(artifact_bucket_blob_path, 'clip_knn.parquet')
        centroid_retrieval_blob_path = os.path.join(artifact_bucket_blob_path,
                                                    'clip_centroid.parquet')

        # Read manifest
        with open(data_manifest_path) as f:
            manifest_load = json.load(f)
        data_manifest = DataManifest.from_dict(manifest_load)
        embedding_path = data_manifest.associated_data.embedding['cf']

        # Copy embeddings to local directory
        storage_helpers.copy_folder_bulk(f"{embedding_path}/*", tmp_embedding_dir_path)
        nb_src_images = len(os.listdir(tmp_embedding_dir_path))
        logging.info('Total number of source images (embedding): %s', nb_src_images)
        centroid_embedding = get_average_embedding(tmp_embedding_dir_path)

        logger.info('A total of %s urls will be requested using the centroid approach.',
                    nb_images_centroid)
        logger.info('A total of %s urls will be requested per individual image (total of %s images)'
                    ' using the knn approach.', nb_images_knn)

        # Setup KNN service
        clip_retrieval_runner = ClipRetrievalLaion5B(
            laion_index_path=laion_indices_folder,
            laion_index_folder=laion_index_folder)
        knn_service = clip_retrieval_runner.setup_knn_service()

        logger.info('Starting centroid clip retrieval')

        # Run clip retrieval with centroid approach
        results_centroid = clip_retrieval_runner.run_query(
            knn_service=knn_service,
            query={'embedding_query': centroid_embedding},
            nb_images_request=nb_images_centroid,
            deduplicate=True,
            benchmark=True)
        logger.info('Centroid clip retrieval complete')

        parquet_helpers.write_clip_retrieval_parquet(
            clip_retrieval_parquet_path=centroid_retrieval_path,
            data_iterable_producer=clip_retrieval_runner.clip_results_producer,
            clip_results=results_centroid)
        del results_centroid

        # Run clip retrieval with KNN approach
        logger.info('Starting knn clip retrieval')
        results_knn = []
        for embedding_file in tqdm(os.listdir(tmp_embedding_dir_path)):
            embedding = np.load(os.path.join(tmp_embedding_dir_path, embedding_file))
            results = clip_retrieval_runner.run_query(knn_service=knn_service,
                                                      query={'embedding_query': embedding},
                                                      nb_images_request=nb_images_knn,
                                                      deduplicate=True,
                                                      benchmark=True)
            results_knn.extend(results)
        logger.info('KNN clip retrieval complete')
        parquet_helpers.write_clip_retrieval_parquet(
            clip_retrieval_parquet_path=knn_retrieval_path,
            data_iterable_producer=clip_retrieval_runner.clip_results_producer,
            clip_results=results_knn)
        del results_knn

        # Upload parquet files to bucket
        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=centroid_retrieval_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=centroid_retrieval_blob_path)

        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=knn_retrieval_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=knn_retrieval_blob_path)

        # Update manifest
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.creation_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        data_manifest.metadata.run_id = run_id

        logger.info('Manifest file created and updated')

        # Write manifest to outputPath
        Path(data_manifest_path_clip_retrieval_component).parent.mkdir(parents=True, exist_ok=True)
        Path(parquet_path_clip_knn_retrieval).parent.mkdir(parents=True, exist_ok=True)
        Path(parquet_path_clip_centroid_retrieval).parent.mkdir(parents=True, exist_ok=True)
        # Write Parquet url files to outputPath
        Path(data_manifest_path_clip_retrieval_component).write_text(data_manifest.to_json())
        Path(parquet_path_clip_knn_retrieval).write_text(
            f'gs://{artifact_bucket}/{knn_retrieval_blob_path}')
        Path(parquet_path_clip_centroid_retrieval).write_text(
            f'gs://{artifact_bucket}/{centroid_retrieval_blob_path}')
        logger.info('Manifest file written to %s', data_manifest_path_clip_retrieval_component)
        logger.info('Job completed.')


if __name__ == '__main__':
    args = parse_args()
    clip_retrieval_component \
        (run_id=args.run_id,
         artifact_bucket=args.artifact_bucket,
         component_name=args.component_name,
         project_id=args.project_id,
         laion_index_url=args.laion_index_url,
         laion_metadata_url=args.laion_metadata_url,
         nb_images_knn=args.nb_images_knn,
         nb_images_centroid=args.nb_images_centroid,
         data_manifest_path=args.data_manifest_path,
         data_manifest_path_clip_retrieval_component=
         args.data_manifest_path_clip_retrieval_component,
         parquet_path_clip_centroid_retrieval=args.parquet_path_clip_centroid_retrieval,
         parquet_path_clip_knn_retrieval=args.parquet_path_clip_knn_retrieval)
