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
load_from_hub_op = comp.load_component(
    "components/load_from_hub/kubeflow_component.yaml"
)
load_from_hub_args = {
    "dataset_name": LoadFromHubConfig.DATASET_NAME,
    "batch_size": LoadFromHubConfig.BATCH_SIZE,
}
load_from_hub_metadata = {
    "run_id": run_id,
    "component_name": load_from_hub_op.__name__,
    "artifact_bucket": artifact_bucket,
}
load_from_hub_args = json.dumps(load_from_hub_args)
load_from_hub_metadata = json.dumps(load_from_hub_metadata)
load_from_hub_spec_path = "components/load_from_hub/load_from_hub.yaml"


# Pipeline
@dsl.pipeline(
    name="image-generator-dataset",
    description="Pipeline that takes example images as input and returns an expanded dataset of "
    "similar images as outputs",
)
# pylint: disable=too-many-arguments, too-many-locals
def sd_dataset_creator_pipeline(
    load_from_hub_args: str = load_from_hub_args,
    load_from_hub_metadata: str = load_from_hub_metadata,
    load_from_hub_spec_path: str = load_from_hub_spec_path,
):

    # Component 1
    load_from_hub_task = load_from_hub_op(
        input_manifest_path="",
        args=load_from_hub_args,
        metadata=load_from_hub_metadata,
        spec_path=load_from_hub_spec_path,
    ).set_display_name("Load initial images")


if __name__ == "__main__":
    compile_and_upload_pipeline(
        pipeline=sd_dataset_creator_pipeline,
        host=KubeflowConfig.HOST,
        env=KubeflowConfig.ENV,
    )