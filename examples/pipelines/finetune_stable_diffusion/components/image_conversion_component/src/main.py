"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""
import os
import json
import argparse
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import pyarrow.dataset as ds
from google.cloud import storage

# pylint: disable=import-error
from helpers import storage_helpers, parquet_helpers, kfp_helpers
from helpers.logger import get_logger
from helpers.manifest_helpers import DataManifest
from utils.img_conversion import KfpPipelineImageConverter


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
    parser.add_argument('--file-extensions',
                        type=list,
                        required=True,
                        help='A list containing the file extensions to convert')
    parser.add_argument('--svg-image-width',
                        type=list,
                        required=True,
                        help='The desired width to scale the converted SVG image to')
    parser.add_argument('--svg-image-height',
                        type=list,
                        required=True,
                        help='The desired height to scale the converted SVG image to')
    parser.add_argument('--data-manifest-path',
                        type=str,
                        required=True,
                        help='The previous component manifest path')
    parser.add_argument('--data-manifest-path-image-conversion-component',
                        type=str,
                        required=True,
                        help='The path to the output manifest file')

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments
def dataset_image_conversion_component(run_id: str,
                                       artifact_bucket: str,
                                       component_name: str,
                                       project_id: str,
                                       file_extensions: list,
                                       data_manifest_path: str,
                                       svg_image_width: int,
                                       svg_image_height: int,
                                       data_manifest_path_image_conversion_component: str) -> None:
    """
    A component that takes a data manifest as input and converts image formats to 'jpeg' which is
    a preferred format for subsequent preprocessing and training steps
    Args:
        run_id (str): The run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifacts
        component_name (str): The name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        file_extensions (list): A list containing the file extensions to convert
        data_manifest_path (str): The previous component manifest path
        svg_image_width (int): the desired width to scale the converted SVG image to
        svg_image_height (int): the desired width to scale the converted SVG image to
        data_manifest_path_image_conversion_component (str): the path to the output manifest file
    """
    # Initialize logger
    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info('Started job...')

    # Parse list variables
    file_extensions = kfp_helpers.parse_kfp_list(file_extensions)

    # Initialize storage client
    storage_client = storage.Client(project=project_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info('created temporary directory %s', tmp_dir)

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition('-')[0]
        artifact_bucket_blob_path = f"custom_artifact/{component_artifact_dir}/{component_name}"
        dataset_parquet_blob_path = os.path.join(artifact_bucket_blob_path, 'dataset.parquet')
        converted_images_blob_path = os.path.join(artifact_bucket_blob_path, 'converted_images')
        converted_images_gcs_uri = f'gs://{artifact_bucket}/{converted_images_blob_path}'
        logger.info("custom artifact will be uploaded to %s",
                    f'gs://{artifact_bucket}/{artifact_bucket_blob_path}')

        # Initialize temporary directories and file paths
        tmp_img_dir_path = os.path.join(tmp_dir, 'img_dir')
        os.makedirs(tmp_img_dir_path, exist_ok=True)
        dataset_parquet_tmp_path = os.path.join(tmp_dir, 'dataset.parquet')
        Path(dataset_parquet_tmp_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info('GCS and temporary paths initialized')

        # Read manifest from previous component
        with open(data_manifest_path) as f:
            manifest_load = json.load(f)
        data_manifest = DataManifest.from_dict(manifest_load)

        # Get index and dataset parquet gcs paths
        index_parquet_prev_gcs_path = data_manifest.index
        # TODO: replace harcoded namespace with string or list input
        dataset_parquet_prev_gcs_path = data_manifest.associated_data.dataset['cf']

        # Download parquet fies locally
        index_parquet_prev_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, index_parquet_prev_gcs_path, tmp_dir)
        dataset_parquet_prev_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, dataset_parquet_prev_gcs_path, tmp_dir)

        # Get index_ids
        index_ids = parquet_helpers.get_column_list_from_parquet(
            parquet_scanner_or_path=index_parquet_prev_tmp_path,
            column_name='index')

        # Initialize parquet dataset
        parquet_dataset = ds.dataset(dataset_parquet_prev_tmp_path) \
            .scanner(columns=['file_extension', 'file_id', 'file_uri', 'file_size'])

        # Start image conversion
        image_converter = KfpPipelineImageConverter(
            parquet_dataset=parquet_dataset,
            updated_parquet_dataset_path=dataset_parquet_tmp_path,
            destination_gcs_uri=converted_images_gcs_uri,
            index_ids=index_ids,
            tmp_img_path=tmp_img_dir_path,
            file_extensions=file_extensions,
            download_list_file_path=os.path.join(tmp_dir, 'download_gcs.txt'),
            upload_list_file_path=os.path.join(tmp_dir, 'upload_gcs.txt'))
        image_converter.start(svg_image_height=svg_image_height, svg_image_width=svg_image_width)

        # Upload newly created parquet file to gcs
        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=dataset_parquet_tmp_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=dataset_parquet_blob_path)

        # Update manifest
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        # TODO: replace harcoded namespace with string or list input
        # TODO: make sure we read all the namespace ('cf', 'laion')
        data_manifest.associated_data.dataset['cf'] = \
            f"gs://{artifact_bucket}/{dataset_parquet_blob_path}"
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.creation_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        data_manifest.metadata.run_id = run_id

        logger.info('Manifest file created and updated')

        # Write manifest to outputPath
        Path(data_manifest_path_image_conversion_component) \
            .parent.mkdir(parents=True, exist_ok=True)
        Path(data_manifest_path_image_conversion_component).write_text(data_manifest.to_json())

        logger.info('Manifest file written to %s', data_manifest_path_image_conversion_component)

        logger.info('Job completed.')


if __name__ == '__main__':
    args = parse_args()
    dataset_image_conversion_component(
        run_id=args.run_id,
        artifact_bucket=args.artifact_bucket,
        component_name=args.component_name,
        project_id=args.project_id,
        file_extensions=args.file_extensions,
        svg_image_width=args.svg_image_width,
        svg_image_height=args.svg_image_height,
        data_manifest_path=args.data_manifest_path,
        data_manifest_path_image_conversion_component=args.
        data_manifest_path_image_conversion_component)
