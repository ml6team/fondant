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
from helpers.logger import get_logger
from helpers import storage_helpers, parquet_helpers, kfp_helpers
from helpers.manifest_helpers import DataManifest
from utils.image_classifier import KfpPipelineImageClassification


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
    parser.add_argument('--clean-cut-classifier-path',
                        type=str,
                        required=True,
                        help='The gcs path where the clean cut classifier is located')
    parser.add_argument('--batch-size-clean-cut',
                        type=int,
                        required=True,
                        help='the batch size of the clean cut classifier')
    parser.add_argument('--data-manifest-path',
                        type=str,
                        required=True,
                        help='The previous component manifest path')
    parser.add_argument('--data-manifest-path-image-classifier-component',
                        type=str,
                        required=True,
                        help='Path to the local file containing the gcs path where the output'
                             ' has been stored')

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
def image_classifier_component(run_id: str,
                               artifact_bucket: str,
                               component_name: str,
                               project_id: str,
                               clean_cut_classifier_path: str,
                               batch_size_clean_cut: int,
                               data_manifest_path: str,
                               data_manifest_path_image_classifier_component: str) -> None:
    """
    Args:
        run_id (str): The run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifact
        component_name (str): the name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        data_manifest_path (str): The previous component manifest path
        clean_cut_classifier_path (str): The gcs path where the clean cut classifier is located
        batch_size_clean_cut (int): the batch size of the clean cut classifier
        data_manifest_path (str): The previous component manifest path
        data_manifest_path_image_classifier_component (str): the current component manifest path
    """

    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info('Started job...')

    # Initialize storage client
    storage_client = storage.Client(project=project_id)

    # Show CUDA availability
    kfp_helpers.get_cuda_availability()

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info('created temporary directory %s', tmp_dir)

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition('-')[0]
        artifact_bucket_blob_path = f"custom_artifact/{component_artifact_dir}/{component_name}"
        index_parquet_blob_path = os.path.join(artifact_bucket_blob_path, 'index.parquet')
        logger.info("custom artifact will be uploaded to %s",
                    f'gs://{artifact_bucket}/{artifact_bucket_blob_path}')

        # Initialize local paths
        tmp_img_dir_path = os.path.join(tmp_dir, 'img_dir')
        clean_cut_classifier_tmp_path = os.path.join(
            tmp_dir, os.path.basename(clean_cut_classifier_path))
        index_parquet_tmp_path = os.path.join(tmp_dir, 'index.parquet')
        os.makedirs(tmp_img_dir_path, exist_ok=True)
        os.makedirs(clean_cut_classifier_tmp_path, exist_ok=True)
        print(clean_cut_classifier_tmp_path)

        # Read manifest
        with open(data_manifest_path) as f:
            manifest_load = json.load(f)
        data_manifest = DataManifest.from_dict(manifest_load)

        # Get index and dataset parquet gcs paths
        index_parquet_prev_gcs_path = data_manifest.index
        datasets_parquet_dict = data_manifest.associated_data.dataset

        # Download index files
        index_parquet_prev_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, index_parquet_prev_gcs_path, tmp_dir)

        # Download namespace datasets
        dataset_parquet_path_dict = {}
        for namespace, dataset_gcs_path in datasets_parquet_dict.items():
            dataset_parquet_tmp_path = storage_helpers.download_file_from_bucket(
                storage_client, dataset_gcs_path, tmp_dir)
            dataset_parquet_path_dict[namespace] = dataset_parquet_tmp_path

        # Get index_ids
        index_ids = parquet_helpers.get_column_list_from_parquet(
            parquet_scanner_or_path=index_parquet_prev_tmp_path,
            column_name='index')

        # Download the clean cut classifier weights
        storage_helpers.copy_folder_bulk(f"{clean_cut_classifier_path}/*",
                                         clean_cut_classifier_tmp_path)

        # Initialize image classifier class
        image_classifier = KfpPipelineImageClassification(
            ccf_classifier_path=clean_cut_classifier_tmp_path,
            clip_model_path='ViT-L-14.pt',
            tmp_path=tmp_dir
        )

        # Classify images for all the available namespaces
        filtered_indexes = []
        for namespace, dataset_parquet_tmp_path in tqdm(dataset_parquet_path_dict.items()):
            logger.info('Starting classification job for for dataset with namespace "%s"',
                        namespace)
            filtered_namespace_index = image_classifier.start(
                index=index_ids,
                parquet_dataset_path=dataset_parquet_tmp_path,
                namespace=namespace,
                batch_size_clean_cut=batch_size_clean_cut)
            filtered_indexes.extend(filtered_namespace_index)

            logger.info('Classification job for %s dataset with namespace "%s" is complete',
                        dataset_parquet_tmp_path, namespace)

        logger.info('Filtering job for all namespaces complete')
        # Estimate the total number of filtered images
        nb_images_before_filtering = len(index_ids)
        nb_images_after_filtering = len(filtered_indexes)
        nb_filtered_image = nb_images_before_filtering - nb_images_after_filtering
        percentage_filtered_images = round(
            100 * (nb_filtered_image / nb_images_before_filtering), 2)

        logger.info(
            "The original number of images was %s. A total of %s images were filtered (%s%%)",
            nb_images_before_filtering, nb_filtered_image, percentage_filtered_images)

        # Upload parquet files
        parquet_helpers.write_index_parquet(
            index_parquet_path=index_parquet_tmp_path,
            data_iterable_producer=lambda id_iterable: (id_element for id_element in id_iterable),
            id_iterable=filtered_indexes)

        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=index_parquet_tmp_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=index_parquet_blob_path)

        # Update manifest
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        data_manifest.index = f"gs://{artifact_bucket}/{index_parquet_blob_path}"
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.creation_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        logger.info('Manifest file updated')

        # Write manifest to outputPath
        Path(data_manifest_path_image_classifier_component).parent.mkdir(parents=True,
                                                                         exist_ok=True)
        Path(data_manifest_path_image_classifier_component).write_text(data_manifest.to_json())

        logger.info('Manifest file written to %s', data_manifest_path_image_classifier_component)

        # Clean up temporary storage
        logger.info('Files removed from temporary storage.')
        logger.info('Job completed.')


if __name__ == '__main__':
    args = parse_args()
    image_classifier_component \
        (run_id=args.run_id,
         artifact_bucket=args.artifact_bucket,
         component_name=args.component_name,
         project_id=args.project_id,
         data_manifest_path=args.data_manifest_path,
         batch_size_clean_cut=args.batch_size_clean_cut,
         clean_cut_classifier_path=args.clean_cut_classifier_path,
         data_manifest_path_image_classifier_component=
         args.data_manifest_path_image_classifier_component)
