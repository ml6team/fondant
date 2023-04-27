"""General pipeline utils"""

import os
import logging
from typing import Callable, Optional

from fondant.import_utils import is_kfp_available

if is_kfp_available():
    import kfp
    from kfp_server_api.exceptions import ApiException

logger = logging.getLogger(__name__)


def compile_and_upload_pipeline(
    pipeline: Callable[[], None], host: str, env: str, pipeline_id: Optional[str] = None
) -> None:
    """Upload pipeline to kubeflow.
    Args:
        pipeline: function that contains the pipeline definition
        host: the url host for kfp
        env: the project run environment (sbx, dev, prd)
        pipeline_id: pipeline id of existing component under the same name. Pass it on when you
        want to delete current existing pipelines
    """
    client = kfp.Client(host=host)

    pipeline_name = f"{pipeline.__name__}:{env}"
    pipeline_filename = f"{pipeline_name}.yaml"
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=pipeline_filename
    )
    if pipeline_id is not None:
        pipeline_versions = client.list_pipeline_versions(pipeline_id).versions
        if pipeline_versions is not None:
            versions_ids = [getattr(version, "id") for version in pipeline_versions]
            for version_id in versions_ids:
                client.delete_pipeline_version(version_id)
            logger.info("Found existing pipeline under the same. Deleting the pipeline")
        client.delete_pipeline(pipeline_id)

    logger.info(f"Uploading pipeline: {pipeline_name}")

    try:
        client.upload_pipeline(pipeline_filename, pipeline_name=pipeline_name)
    except Exception as e:
        raise ApiException(
            f"Failed to upload the pipeline. Make sure that the pipeline {pipeline_name} does"
            f" not exist. If you have a pipeline under a similar name, pass in the `pipeline id`"
            f" in order to delete the existing pipeline"
        ) from e
    os.remove(pipeline_filename)
