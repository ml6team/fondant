"""General pipeline utils"""

import json
import os
import logging
from typing import Callable

from express.import_utils import is_kfp_available

if is_kfp_available():
    import kfp

logger = logging.getLogger(__name__)


def create_extra_args(**kwargs):
    extra_args = json.dumps(**kwargs)
    return extra_args


def create_metadata_args(component, artifact_bucket):
    run_id = "{{workflow.name}}"

    metadata_args = {
        "run_id": run_id,
        "component_name": component.__name__,
        "artifact_bucket": artifact_bucket,
    }
    metadata_args = json.dumps(metadata_args)
    return metadata_args


def compile_and_upload_pipeline(
    pipeline: Callable[[], None], host: str, env: str
) -> None:
    """Upload pipeline to kubeflow.
    Args:
        pipeline (Callable): function that contains the pipeline definition
        host (str): the url host for kfp
        env (str): the project run environment (sbx, dev, prd)
    """
    client = kfp.Client(host=host)

    pipeline_name = f"{pipeline.__name__}:{env}"
    pipeline_filename = f"{pipeline_name}.yaml"
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=pipeline_filename
    )

    existing_pipelines = client.list_pipelines(page_size=100).pipelines
    for existing_pipeline in existing_pipelines:
        if existing_pipeline.name == pipeline_name:
            # Delete existing pipeline before uploading
            logger.warning(
                f"Pipeline {pipeline_name} already exists. Deleting old pipeline..."
            )
            client.delete_pipeline_version(existing_pipeline.default_version.id)
            client.delete_pipeline(existing_pipeline.id)

    logger.info(f"Uploading pipeline: {pipeline_name}")
    client.upload_pipeline(pipeline_filename, pipeline_name=pipeline_name)
    os.remove(pipeline_filename)
