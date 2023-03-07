"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""
import os
import argparse
import logging
import json
import tempfile
from pathlib import Path
from datetime import datetime

import pyarrow.compute as pc
from google.cloud import storage

# pylint: disable=import-error
from helpers.logger import get_logger
from helpers import storage_helpers, parquet_helpers, kfp_helpers
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
    parser.add_argument('--min-file-size',
                        type=int,
                        required=True,
                        help='The minimum size of an image (filter)')
    parser.add_argument('--max-file-size',
                        type=int,
                        required=True,
                        help='The maximum size of an image (filter)')
    parser.add_argument('--image-formats',
                        type=list,
                        required=True,
                        help='The image formats to keep (filter)')
    parser.add_argument('--data-manifest-path',
                        type=str,
                        required=True,
                        help='The previous component manifest path')
    parser.add_argument('--data-manifest-path-filter-component',
                        type=str,
                        required=True,
                        help='The path to the output manifest file')

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments
def image_filter_component(run_id: str,
                           artifact_bucket: str,
                           component_name: str,
                           project_id: str,
                           min_file_size: int,
                           max_file_size: int,
                           image_formats: list,
                           data_manifest_path: str,
                           data_manifest_path_filter_component: str) -> None:
    """
    A component that takes a data manifest as an input and filters it according to metadata related
    information.
    Args:
        run_id (str): The run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifacts
        component_name (str): The name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        min_file_size (int): The minimum size of an image (filter)
        max_file_size (int): The maximum size of an image (filter)
        image_formats (list): The image formats to keep (filter)
        data_manifest_path (str): The previous component manifest path
        data_manifest_path_filter_component (str): the path to the output manifest file
    """

    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info('Started job...')

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info('created temporary directory %s', tmp_dir)
        # Parse list variables
        image_formats = kfp_helpers.parse_kfp_list(image_formats)

        # Initialize storage client
        storage_client = storage.Client(project=project_id)

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition('-')[0]
        artifact_bucket_blob_path = f"custom_artifact/{component_artifact_dir}/{component_name}"
        index_parquet_blob_path = os.path.join(artifact_bucket_blob_path, 'index.parquet')
        logger.info("custom artifact will be uploaded to %s",
                    f'gs://{artifact_bucket}/{artifact_bucket_blob_path}')

        index_parquet_tmp_path = os.path.join(tmp_dir, 'index.parquet')
        Path(index_parquet_tmp_path).parent.mkdir(parents=True, exist_ok=True)

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

        # Get indices
        index_before_filtering = parquet_helpers.get_column_list_from_parquet(
            parquet_scanner_or_path=index_parquet_prev_tmp_path,
            column_name='index')
        # Construct parquet filters and filter based on the criteria
        filters = (pc.field("file_id").isin(index_before_filtering)) & \
                  (pc.field("file_size") > pc.scalar(min_file_size)) & \
                  (pc.field("file_size") < pc.scalar(max_file_size)) & \
                  (pc.field("file_extension").isin(image_formats))

        filtered_dataset_scanner = parquet_helpers.filter_parquet_file(
            file_path=dataset_parquet_prev_tmp_path,
            filters=filters)

        # Write new index ids parquet file and upload it to gcs
        index_after_filtering = parquet_helpers.get_column_list_from_parquet(
            parquet_scanner_or_path=filtered_dataset_scanner,
            column_name='file_id')

        parquet_helpers.write_index_parquet(
            index_parquet_path=index_parquet_tmp_path,
            data_iterable_producer=lambda id_iterable: (id_element for id_element in id_iterable),
            id_iterable=index_after_filtering)

        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=index_parquet_tmp_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=index_parquet_blob_path)

        # Estimate the total number of filtered images
        nb_images_before_filtering = len(index_before_filtering)
        nb_images_after_filtering = parquet_helpers.get_nb_rows_from_parquet(index_parquet_tmp_path)
        nb_filtered_image = nb_images_before_filtering - nb_images_after_filtering
        percentage_filtered_images = round(
            100 * (nb_filtered_image / nb_images_before_filtering), 2)

        logger.info(
            "The original number of images was %s. A total of %s images were filtered (%s%%)",
            nb_images_before_filtering, nb_filtered_image, percentage_filtered_images)

        # Update manifest
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        data_manifest.index = f"gs://{artifact_bucket}/{index_parquet_blob_path}"
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.creation_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        logger.info('Manifest file updated')

        # Write manifest to outputPath
        Path(data_manifest_path_filter_component).parent.mkdir(parents=True, exist_ok=True)
        Path(data_manifest_path_filter_component).write_text(data_manifest.to_json())

        logger.info('Manifest file written to %s', data_manifest_path_filter_component)

        # Clean up temporary storage
        logger.info('Files removed from temporary storage.')
        logger.info('Job completed.')


if __name__ == '__main__':
    args = parse_args()
    image_filter_component(
        run_id=args.run_id,
        artifact_bucket=args.artifact_bucket,
        component_name=args.component_name,
        project_id=args.project_id,
        max_file_size=args.max_file_size,
        min_file_size=args.min_file_size,
        image_formats=args.image_formats,
        data_manifest_path=args.data_manifest_path,
        data_manifest_path_filter_component=args.data_manifest_path_filter_component)
