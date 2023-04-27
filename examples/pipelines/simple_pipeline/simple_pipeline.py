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

# Load Components
base_path = KubeflowConfig.BASE_PATH
run_id = "{{workflow.name}}"

# Component 1
load_from_hub_op = comp.load_component("components/load_from_hub/kubeflow_component.yaml")
# Component 2
image_filtering_op = comp.load_component("components/image_filtering/kubeflow_component.yaml")

load_from_hub_metadata = {
    "base_path": base_path,
    "run_id": run_id,
    "component_id": load_from_hub_op.__name__,
}
image_filtering_metadata = {
    "base_path": base_path,
    "run_id": run_id,
    "component_id": image_filtering_op.__name__,
}
load_from_hub_metadata = json.dumps(load_from_hub_metadata)
image_filtering_metadata = json.dumps(image_filtering_metadata)


# Pipeline
@dsl.pipeline(
    name="simple-pipeline",
    description="Simple pipeline that takes example images as input and embeds them using CLIP",
)
# pylint: disable=too-many-arguments, too-many-locals
def simple_pipeline(
    load_from_hub_dataset_name: str = LoadFromHubConfig.DATASET_NAME,
    load_from_hub_batch_size: int = LoadFromHubConfig.BATCH_SIZE,
    load_from_hub_metadata: str = load_from_hub_metadata,
    image_filter_min_width: int = ImageFilterConfig.MIN_WIDTH,
    image_filter_min_height: int = ImageFilterConfig.MIN_HEIGHT,
    image_filtering_metadata: str = image_filtering_metadata,
):

    # Component 1
    load_from_hub_task = load_from_hub_op(
        dataset_name=load_from_hub_dataset_name,
        batch_size=load_from_hub_batch_size,
        metadata=load_from_hub_metadata,
    ).set_display_name("Load initial images")

    # Component 2
    image_filter_task = image_filtering_op(
        input_manifest_path=load_from_hub_task.outputs["output_manifest_path"],
        min_width=image_filter_min_width,
        min_height=image_filter_min_height,
        metadata=image_filtering_metadata,
    ).set_display_name("Filter images")


if __name__ == "__main__":
    compile_and_upload_pipeline(
        pipeline=simple_pipeline,
        host=KubeflowConfig.HOST,
        env=KubeflowConfig.ENV,
        pipeline_id="b02dd7e3-445e-49a1-8854-7cf01d6abc19",
    )