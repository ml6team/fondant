"""Pipeline used to create a stable diffusion dataset from a set of given images."""
# pylint: disable=import-error
import json

from kfp import components as comp
from kfp import dsl

from config.general_config import KubeflowConfig
from config.components_config import (
    LoadFromHubConfig,
)
from express.pipeline_utils import compile_and_upload_pipeline

# Load Components
run_id = "{{workflow.name}}"
artifact_bucket = KubeflowConfig.ARTIFACT_BUCKET

# Component 1
load_from_hub_op = comp.load_component("components/load_from_hub/kubeflow_component.yaml")

load_from_hub_metadata = {
    "base_path": artifact_bucket,
    "run_id": run_id,
    "component_id": load_from_hub_op.__name__,
}
load_from_hub_metadata = json.dumps(load_from_hub_metadata)


# Pipeline
@dsl.pipeline(
    name="image-generator-dataset",
    description="Pipeline that takes example images as input and returns an expanded dataset of "
    "similar images as outputs",
)
# pylint: disable=too-many-arguments, too-many-locals
def sd_dataset_creator_pipeline(
    load_from_hub_dataset_name: str = LoadFromHubConfig.DATASET_NAME,
    load_from_hub_batch_size: int = LoadFromHubConfig.BATCH_SIZE,
    load_from_hub_metadata: str = load_from_hub_metadata,
):

    # Component 1
    load_from_hub_task = load_from_hub_op(
        dataset_name=load_from_hub_dataset_name,
        batch_size=load_from_hub_batch_size,
        metadata=load_from_hub_metadata,
    ).set_display_name("Load initial images")


if __name__ == "__main__":
    compile_and_upload_pipeline(
        pipeline=sd_dataset_creator_pipeline,
        host=KubeflowConfig.HOST,
        env=KubeflowConfig.ENV,
    )