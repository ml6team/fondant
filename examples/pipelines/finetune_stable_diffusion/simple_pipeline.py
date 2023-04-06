"""Pipeline used to create a stable diffusion dataset from a set of given images."""
# pylint: disable=import-error
import json

from kfp import components as comp
from kfp import dsl

from kubernetes import client as k8s_client

from config.general_config import KubeflowConfig
from config.components_config import (
    LoadFromHubConfig,
    ImageFilterConfig,
    EmbeddingConfig,
    ClipRetrievalConfig,
)
from express.pipeline_utils import compile_and_upload_pipeline

# Load Components
run_id = "{{workflow.name}}"
artifact_bucket = KubeflowConfig.ARTIFACT_BUCKET

# Component 1
load_from_hub_op = comp.load_component(
    "components/load_from_hub_component/component.yaml"
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

# Component 2
image_filter_op = comp.load_component(
    "components/image_filter_component/component.yaml"
)
image_filter_args = {
    "min_height": ImageFilterConfig.MIN_HEIGHT,
    "min_width": ImageFilterConfig.MIN_WIDTH,
}
image_filter_metadata = {
    "run_id": run_id,
    "component_name": image_filter_op.__name__,
    "artifact_bucket": artifact_bucket,
}
image_filter_args = json.dumps(image_filter_args)
image_filter_metadata = json.dumps(image_filter_metadata)


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
    image_filter_args: str = image_filter_args,
    image_filter_metadata: str = image_filter_metadata,
):

    # Component 1
    load_from_hub_task = load_from_hub_op(
        args=load_from_hub_args,
        metadata=load_from_hub_metadata,
    ).set_display_name("Load initial images")

    # # Component 2
    # image_filter_task = image_filter_op(
    #     args=image_filter_args,
    #     metadata=image_filter_metadata,
    #     input_manifest=load_from_hub_task.outputs["output_manifest"],
    # ).set_display_name("Filter images")


if __name__ == "__main__":
    compile_and_upload_pipeline(
        pipeline=sd_dataset_creator_pipeline,
        host=KubeflowConfig.HOST,
        env=KubeflowConfig.ENV,
    )
