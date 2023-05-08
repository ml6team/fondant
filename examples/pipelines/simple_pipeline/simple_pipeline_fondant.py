"""Pipeline used to create a stable diffusion dataset from a set of given images."""
from config.general_config import KubeflowConfig
from config.components_config import (
    LoadFromHubConfig,
    ImageFilterConfig,
)

from fondant.pipeline import FondantComponentOp, FondantPipeline, FondantClient

# General configs
pipeline_name = "Test fondant pipeline"
pipeline_description = "A test pipeline"
pipeline_host = KubeflowConfig.HOST
pipeline_base_path = KubeflowConfig.BASE_PATH

client = FondantClient(host=pipeline_host)

# Define component ops
load_from_hub_op = FondantComponentOp(
    component_spec_path="components/load_from_hub/fondant_component.yaml",
    arguments={"dataset_name": LoadFromHubConfig.DATASET_NAME},
)

image_filtering_op = FondantComponentOp(
    component_spec_path="components/image_filtering/fondant_component.yaml",
    arguments={
        "min_width": ImageFilterConfig.MIN_WIDTH,
        "min_height": ImageFilterConfig.MIN_HEIGHT,
    },
)

pipeline = FondantPipeline(pipeline_name=pipeline_name, base_path=pipeline_base_path)

pipeline.add_op(load_from_hub_op)
pipeline.add_op(image_filtering_op, dependency=load_from_hub_op)

client.compile_and_upload(pipeline=pipeline)
