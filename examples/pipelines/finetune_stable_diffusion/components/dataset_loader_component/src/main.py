"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""
import os
import argparse
import logging
import tempfile
from pathlib import Path
from datetime import datetime

from google.cloud import storage

# pylint: disable=import-error
from helpers import storage_helpers, parquet_helpers
from helpers.logger import get_logger
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
    parser.add_argument('--source-dataset-bucket',
                        type=str,
                        required=True,
                        help='The GCS bucket containing the dataset to load')
    parser.add_argument('--source-dataset-blob',
                        type=str,
                        required=True,
                        help='The GCS blob withing the specified bucket containing the dataset'
                             ' to load')
    parser.add_argument('--namespace',
                        type=str,
                        required=True,
                        help='The dataset namespace (abbreviation for data source)')
    parser.add_argument('--data-manifest-path',
                        type=str,
                        required=True,
                        help='The data manifest output artifact')
    return parser.parse_args()


# pylint: disable=too-many-locals, too-many-arguments
def dataset_loader_component(run_id: str,
                             artifact_bucket: str,
                             component_name: str,
                             project_id: str,
                             source_dataset_bucket: str,
                             source_dataset_blob: str,
                             namespace: str,
                             data_manifest_path: str) -> None:
    """
    A component that takes an images dataset as input and initializes a data manifest
    and writes it to an output file.
    Args:
        run_id (str): the run id of the pipeline
        artifact_bucket (str): The GCS bucket used to store the artifacts
        component_name (str): the name of the component (used to create gcs artefact path)
        project_id (str): The id of the gcp-project
        source_dataset_bucket (str): The GCS bucket containing the dataset to load
        source_dataset_blob (str): The GCS blob withing the specified bucket containing the dataset
         to load
        namespace (str): The dataset namespace (abbreviation for data source)
        data_manifest_path (str): the path to write the manifest
    """
    logger = get_logger(name=__name__, level=logging.INFO)
    logger.info('Started job...')
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info('created temporary directory %s', tmp_dir)

        # Initialize storage client
        storage_client = storage.Client(project=project_id)

        # Initialize GCS temporary storage paths
        # Parse to match the directory convention from MINIO (<pipeline_component_name>-<run-uid>)
        component_artifact_dir = run_id.rpartition('-')[0]
        artifact_bucket_blob_path = f"custom_artifact/{component_artifact_dir}/{component_name}"

        logger.info("custom artifact will be uploaded to %s",
                    f'gs://{artifact_bucket}/{artifact_bucket_blob_path}')

        dataset_parquet_tmp_path = os.path.join(tmp_dir, 'dataset.parquet')
        dataset_parquet_blob_path = os.path.join(artifact_bucket_blob_path, 'dataset.parquet')
        index_parquet_tmp_path = os.path.join(tmp_dir, 'index.parquet')
        index_parquet_blob_path = os.path.join(artifact_bucket_blob_path, 'index.parquet')
        Path(dataset_parquet_tmp_path).parent.mkdir(parents=True, exist_ok=True)
        Path(index_parquet_tmp_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info('GCS and temporary paths initialized')

        # Write parquet index file
        parquet_helpers.write_index_parquet(index_parquet_path=index_parquet_tmp_path,
                                            data_iterable_producer=storage_helpers.get_blob_id,
                                            storage_client=storage_client,
                                            bucket_name=source_dataset_bucket,
                                            prefix=source_dataset_blob,
                                            id_prefix=namespace)
        # Write parquet dataset
        parquet_helpers.write_dataset_parquet \
            (dataset_parquet_path=dataset_parquet_tmp_path,
             data_iterable_producer=storage_helpers.get_blob_metadata,
             storage_client=storage_client,
             bucket_name=source_dataset_bucket,
             prefix=source_dataset_blob,
             id_prefix=namespace)

        logger.info('Parquet manifest files updated')

        # Upload the parquet
        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=dataset_parquet_tmp_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=dataset_parquet_blob_path)
        storage_helpers.upload_file_to_bucket(storage_client=storage_client,
                                              file_to_upload_path=index_parquet_tmp_path,
                                              bucket_name=artifact_bucket,
                                              blob_path=index_parquet_blob_path)
        logger.info('Parquet manifest files uploaded to GCS')

        data_manifest = DataManifest()
        data_manifest.dataset_id = f"{run_id}_{component_name}"
        data_manifest.index = f'gs://{artifact_bucket}/{index_parquet_blob_path}'
        data_manifest.associated_data.dataset[namespace] = \
            f'gs://{artifact_bucket}/{dataset_parquet_blob_path}'
        data_manifest.metadata.branch = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.commit_hash = ""  # TODO: Fill from docker build env var
        data_manifest.metadata.creation_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        data_manifest.metadata.run_id = run_id

        logger.info('Manifest file created and updated')

        # Write manifest to outputPath
        Path(data_manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(data_manifest_path).write_text(data_manifest.to_json())

        logger.info('Manifest file written to %s', data_manifest_path)

        # Clean up temporary storage
        logger.info('Files removed from temporary storage.')
        logger.info('Job completed.')


if __name__ == '__main__':
    args = parse_args()
    dataset_loader_component(run_id=args.run_id,
                             artifact_bucket=args.artifact_bucket,
                             component_name=args.component_name,
                             project_id=args.project_id,
                             source_dataset_bucket=args.source_dataset_bucket,
                             source_dataset_blob=args.source_dataset_blob,
                             namespace=args.namespace,
                             data_manifest_path=args.data_manifest_path)
