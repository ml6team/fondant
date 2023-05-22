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
from utils.image_captioning import KfpPipelineImageCaptioner
from utils.blip_model import BlipModel


def parse_args():
    """Parse component arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", type=str, required=True, help="The run id of the pipeline"
    )
    parser.add_argument(
        "--artifact-bucket",
        type=str,
        required=True,
        help="The GCS bucket used to store the artifacts",
    )
    parser.add_argument(
        "--component-name", type=str, required=True, help="The name of the component"
    )
    parser.add_argument(
        "--project-id", type=str, required=True, help="The id of the gcp-project"
    )
    parser.add_argument(
        "--min-length", type=int, required=True, help="The minimum caption length"
    )
    parser.add_argument(
        "--max-length", type=int, required=True, help="The maximum caption length"
    )
    parser.add_argument(
        "--beams", type=int, required=True, help="The blip beam parameters"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="The batch size to pass to the captioning component",
    )
    parser.add_argument(
        "--data-manifest-path",
        type=str,
        required=True,
        help="The previous component manifest path",
    )
    parser.add_argument(
        "--data-manifest-path-caption-component",
        type=str,
        required=True,
        help="The path to the output manifest file",
    )

    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments
def dataset_caption_component(
    run_id: str,
    artifact_bucket: str,
    component_name: str,
    project_id: str,
    min_length: int,
    max_length: int,
    batch_size: int,
    beams: int,
    data_manifest_path: str,
    data_manifest_path_caption_component: str,
) -> None:
    """
    A component that takes an images dataset as input and initializes a data manifest
    and writes it to an output file.
    Args:
        run_id (str): the run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifacts
        component_name (str): the name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        min_length (str): The minimum caption length
        max_length (str): the maximum caption length
        batch_size (int): The batch size of images to pass to the blip model
        beams (int): The blip beam parameters
        data_manifest_path (str): The previous component manifest path
        data_manifest_path_caption_component (str): the path to the output manifest file
    """
    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info("Started job...")

    # Show CUDA availability
    kfp_helpers.get_cuda_availability()

    # Initialize storage client
    storage_client = storage.Client(project=project_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("created temporary directory %s", tmp_dir)

        # Initialize GCS custom artifact path
        component_artifact_dir = run_id.rpartition("-")[0]
        artifact_bucket_blob_path = (
            f"custom_artifact/{component_artifact_dir}/{component_name}"
        )
        logger.info(
            "custom artifact will be uploaded to %s",
            f"gs://{artifact_bucket}/{artifact_bucket_blob_path}",
        )

        tmp_img_dir_path = os.path.join(tmp_dir, "img_dir")
        os.makedirs(tmp_img_dir_path, exist_ok=True)

        # Read manifest
        with open(data_manifest_path) as f:
            manifest_load = json.load(f)
        data_manifest = DataManifest.from_dict(manifest_load)
        # Get index and dataset parquet gcs paths
        index_parquet_prev_gcs_path = data_manifest.index
        datasets_parquet_dict = data_manifest.associated_data.dataset

        # Download index files
        index_parquet_prev_tmp_path = storage_helpers.download_file_from_bucket(
            storage_client, index_parquet_prev_gcs_path, tmp_dir
        )

        # Download namespace datasets
        dataset_parquet_path_dict = {}
        for namespace, dataset_gcs_path in datasets_parquet_dict.items():
            dataset_parquet_tmp_path = storage_helpers.download_file_from_bucket(
                storage_client, dataset_gcs_path, tmp_dir
            )
            dataset_parquet_path_dict[namespace] = dataset_parquet_tmp_path

        # Initialize Blip model
        blip_model = BlipModel(model_path="model*_base_caption.pth")

        # Get index_ids
        index_ids_images_to_caption = parquet_helpers.get_column_list_from_parquet(
            parquet_scanner_or_path=index_parquet_prev_tmp_path, column_name="index"
        )

        # Construct parquet filters and filter based on the criteria
        filters = pc.field("file_id").isin(index_ids_images_to_caption)

        # Caption images for all the available namespaces
        for namespace, dataset_parquet_tmp_path in dataset_parquet_path_dict.items():
            logger.info(
                'Starting caption and upload job for for dataset with namespace "%s"',
                namespace,
            )

            parquet_dataset_scanner = parquet_helpers.filter_parquet_file(
                file_path=dataset_parquet_tmp_path,
                filters=filters,
                batch_size=batch_size,
            )

            # Caption images and store them in a parquet file
            kfp_image_captioner = KfpPipelineImageCaptioner(
                blip_model=blip_model,
                parquet_dataset=parquet_dataset_scanner,
                tmp_img_path=tmp_img_dir_path,
                tmp_dir=tmp_dir,
                namespace=namespace,
            )

            kfp_image_captioner.start(
                min_length=min_length, max_length=max_length, beams=beams
            )

            captions_parquet_tmp_path = kfp_image_captioner.get_caption_path()

            # Upload caption parquet file to bucket bucket
            captions_parquet_blob_path = os.path.join(
                artifact_bucket_blob_path, f"captions_{namespace}.parquet"
            )
            storage_helpers.upload_file_to_bucket(
                storage_client=storage_client,
                file_to_upload_path=captions_parquet_tmp_path,
                bucket_name=artifact_bucket,
                blob_path=captions_parquet_blob_path,
            )
            logger.info(
                'Caption and upload job for dataset with namespace "%s" is complete',
                namespace,
            )

            # Update manifest
            data_manifest.associated_data.caption[
                namespace
            ] = f"gs://{artifact_bucket}/{captions_parquet_blob_path}"

        # Update manifest
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.creation_date = datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S"
        )
        data_manifest.metadata.run_id = run_id

        logger.info("Manifest file updated")

        # Write manifest to outputPath
        Path(data_manifest_path_caption_component).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(data_manifest_path_caption_component).write_text(data_manifest.to_json())

        logger.info("Manifest file written to %s", data_manifest_path_caption_component)

        logger.info("Job completed.")


if __name__ == "__main__":
    args = parse_args()
    dataset_caption_component(
        run_id=args.run_id,
        artifact_bucket=args.artifact_bucket,
        component_name=args.component_name,
        project_id=args.project_id,
        min_length=args.min_length,
        max_length=args.max_length,
        batch_size=args.batch_size,
        beams=args.beams,
        data_manifest_path=args.data_manifest_path,
        data_manifest_path_caption_component=args.data_manifest_path_caption_component,
    )
