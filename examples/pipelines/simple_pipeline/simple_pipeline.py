"""Pipeline used to create a stable diffusion dataset from a set of given images."""
# pylint: disable=import-error
import json

from kfp import components as comp
from kfp import dsl

from config.general_config import KubeflowConfig
from config.components_config import (
    LoadFromHubConfig,
    ImageFilterConfig,
)
from fondant.pipeline_utils import compile_and_upload_pipeline

# Define metadata
base_path = KubeflowConfig.BASE_PATH
run_id = "{{workflow.name}}"
metadata = json.dumps({"base_path": base_path, "run_id": run_id})

# Define component ops
load_from_hub_op = comp.load_component("components/load_from_hub/kubeflow_component.yaml")
image_filtering_op = comp.load_component("components/image_filtering/kubeflow_component.yaml")


# Pipeline
@dsl.pipeline(
    name="simple-pipeline",
    description="Simple pipeline that takes example images as input and embeds them using CLIP",
)
# pylint: disable=too-many-arguments, too-many-locals
def simple_pipeline(
    metadata: str = metadata,
    load_from_hub_dataset_name: str = LoadFromHubConfig.DATASET_NAME,
    load_from_hub_batch_size: int = LoadFromHubConfig.BATCH_SIZE,
    image_filter_min_width: int = ImageFilterConfig.MIN_WIDTH,
    image_filter_min_height: int = ImageFilterConfig.MIN_HEIGHT,
):

    # Component 1
    load_from_hub_task = load_from_hub_op(
        dataset_name=load_from_hub_dataset_name,
        batch_size=load_from_hub_batch_size,
        metadata=metadata,
    ).set_display_name("Load initial images")

    # Component 2
    image_filter_task = image_filtering_op(
        input_manifest_path=load_from_hub_task.outputs["output_manifest_path"],
        min_width=image_filter_min_width,
        min_height=image_filter_min_height,
    ).set_display_name("Filter images")


if __name__ == "__main__":
    compile_and_upload_pipeline(
        pipeline=simple_pipeline,
        host=KubeflowConfig.HOST,
        env=KubeflowConfig.ENV,
        pipeline_id="f0216368-d972-423d-94bb-d29ec4274f7a",
    )