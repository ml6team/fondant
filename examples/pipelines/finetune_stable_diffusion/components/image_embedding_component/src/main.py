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

import pyarrow.compute as pc
from google.cloud import storage

# pylint: disable=import-error
from helpers.logger import get_logger
from helpers import storage_helpers, parquet_helpers, kfp_helpers
from helpers.manifest_helpers import DataManifest
from utils.image_embedding import KfpPipelineImageEmbedder


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
    parser.add_argument('--batch-size',
                        type=int,
                        required=True,
                        help='The number of images to batch before embedding')
    parser.add_argument('--data-manifest-path',
                        type=str,
                        required=True,
                        help='The previous component manifest path')
    parser.add_argument('--data-manifest-path-embedding-component',
                        type=str,
                        required=True,
                        help='The path to the output manifest file')

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments
def image_embedding_component(run_id: str,
                              artifact_bucket: str,
                              component_name: str,
                              project_id: str,
                              batch_size: int,
                              data_manifest_path: str,
                              data_manifest_path_embedding_component: str) -> None:
    """
    A component that takes an images dataset as input and generated image embeddings out of them
    Args:
        run_id (str): the run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifacts
        component_name (str): the name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        batch_size (int): the number of images to batch before embedding
        data_manifest_path (str): The previous component manifest path
        data_manifest_path_embedding_component (str): the path to the output manifest file
    """
    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info('Started job...')

    # Show CUDA availability
    kfp_helpers.get_cuda_availability()

    # Initialize storage client
    storage_client = storage.Client(project=project_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info('created temporary directory %s', tmp_dir)

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition('-')[0]
        artifact_bucket_blob_path = f"custom_artifact/{component_artifact_dir}/{component_name}"
        embedding_blob_path = f"gs://{artifact_bucket}/{artifact_bucket_blob_path}/embeddings"
        logger.info("custom artifact will be uploaded to %s",
                    f'gs://{artifact_bucket}/{artifact_bucket_blob_path}')

        tmp_img_dir_path = os.path.join(tmp_dir, 'img_dir')
        tmp_embedding_dir_path = os.path.join(tmp_dir, 'embeddings_dir')
        os.makedirs(tmp_img_dir_path, exist_ok=True)
        os.makedirs(tmp_embedding_dir_path, exist_ok=True)
        dataset_id_parquet_tmp_path = os.path.join(tmp_dir, 'dataset.parquet')
        Path(dataset_id_parquet_tmp_path).parent.mkdir(parents=True, exist_ok=True)

        # Read manifest
        with open(data_manifest_path) as f:
            manifest_load = json.load(f)
        data_manifest = DataManifest.from_dict(manifest_load)

        # Get index and dataset parquet gcs paths
        index_parquet_prev_gcs_path = data_manifest.index
        # TODO: replace harcoded namespace
        dataset_parquet_prev_gcs_path = data_manifest.associated_data.dataset['cf']

        # Download parquet files locally
        index_parquet_prev_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, index_parquet_prev_gcs_path, tmp_dir)
        dataset_parquet_prev_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, dataset_parquet_prev_gcs_path, tmp_dir)

        # Get index_ids
        index_ids_images_to_embed = parquet_helpers.get_column_list_from_parquet(
            parquet_scanner_or_path=index_parquet_prev_tmp_path,
            column_name='index')

        # Construct parquet filters and filter based on the criteria
        filters = (pc.field("file_id").isin(index_ids_images_to_embed))

        filtered_dataset_scanner = parquet_helpers.filter_parquet_file(
            file_path=dataset_parquet_prev_tmp_path,
            filters=filters,
            batch_size=batch_size)

        # Caption images and store them in a parquet file
        kfp_image_embedder = KfpPipelineImageEmbedder(
            parquet_dataset=filtered_dataset_scanner,
            embedding_blob_path=embedding_blob_path,
            tmp_img_path=tmp_img_dir_path,
            tmp_embedding_path=tmp_embedding_dir_path,
            download_list_file_path=os.path.join(tmp_dir, 'download_gcs.txt')
        )

        kfp_image_embedder.start()

        # Update manifest
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        # TODO: replace harcoded namespace with string or list input
        data_manifest.associated_data.embedding['cf'] = embedding_blob_path
        data_manifest.metadata.creation_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        data_manifest.metadata.run_id = run_id

        logger.info('Manifest file created and updated')

        # Write manifest to outputPath
        Path(data_manifest_path_embedding_component).parent.mkdir(parents=True, exist_ok=True)
        Path(data_manifest_path_embedding_component).write_text(data_manifest.to_json())

        logger.info('Manifest file written to %s', data_manifest_path_embedding_component)

        logger.info('Job completed.')


if __name__ == '__main__':
    args = parse_args()
    image_embedding_component \
        (run_id=args.run_id,
         artifact_bucket=args.artifact_bucket,
         component_name=args.component_name,
         project_id=args.project_id,
         batch_size=args.batch_size,
         data_manifest_path=args.data_manifest_path,
         data_manifest_path_embedding_component=args.data_manifest_path_embedding_component)
