"""Pipeline used to create a stable diffusion dataset from a set of initial prompts."""
# pylint: disable=import-error
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import (
    ComponentOp,
    Pipeline,
    Client,
)
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
# General configs
pipeline_name = "controlnet-pipeline"
pipeline_description = "Pipeline that collects data to train ControlNet"

client = Client(host=PipelineConfigs.HOST)

# Define component ops
generate_prompts_op = ComponentOp(
    component_spec_path="components/generate_prompts/fondant_component.yaml"
)
laion_retrieval_op = ComponentOp.from_registry(
    name="prompt_based_laion_retrieval",
    arguments={"num_images": 2, "aesthetic_score": 9, "aesthetic_weight": 0.5},
)
download_images_op = ComponentOp(
    component_spec_path="components/download_images/fondant_component.yaml",
    arguments={
        "timeout": 10,
        "retries": 0,
        "image_size": 512,
        "resize_mode": "center_crop",
        "resize_only_if_bigger": False,
        "min_image_size": 0,
        "max_aspect_ratio": 2.5,
    },
)
caption_images_op = ComponentOp(
    component_spec_path="components/caption_images/fondant_component.yaml",
    arguments={
        "model_id": "microsoft/git-base-coco",
        "batch_size": 2,
        "max_new_tokens": 50,
    },
    number_of_gpus=1,
    node_pool_name="model-inference-pool",
)

pipeline = Pipeline(pipeline_name=pipeline_name, base_path=PipelineConfigs.BASE_PATH)

pipeline.add_op(generate_prompts_op)
pipeline.add_op(laion_retrieval_op, dependencies=generate_prompts_op)
pipeline.add_op(download_images_op, dependencies=laion_retrieval_op)
pipeline.add_op(caption_images_op, dependencies=download_images_op)

client.compile_and_run(pipeline=pipeline)
