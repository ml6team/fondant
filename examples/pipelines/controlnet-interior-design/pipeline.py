"""Pipeline used to create a stable diffusion dataset from a set of initial prompts."""
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)
# General configs
pipeline_name = "controlnet-pipeline"
pipeline_description = "Pipeline that collects data to train ControlNet"

# Define component ops
generate_prompts_op = ComponentOp(
    component_spec_path="components/generate_prompts/fondant_component.yaml",
    arguments={"n_rows_to_load": None},
)
laion_retrieval_op = ComponentOp.from_registry(
    name="prompt_based_laion_retrieval",
    arguments={
        "num_images": 2,
        "aesthetic_score": 9,
        "aesthetic_weight": 0.5,
        "url": None,
    },
)
download_images_op = ComponentOp.from_registry(
    name="download_images",
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
caption_images_op = ComponentOp.from_registry(
    name="caption_images",
    arguments={
        "model_id": "Salesforce/blip-image-captioning-base",
        "batch_size": 2,
        "max_new_tokens": 50,
    },
    number_of_gpus=1,
    node_pool_name="model-inference-pool",
)
segment_images_op = ComponentOp.from_registry(
    name="segment_images",
    arguments={
        "model_id": "openmmlab/upernet-convnext-small",
        "batch_size": 2,
    },
    number_of_gpus=1,
    node_pool_name="model-inference-pool",
)

write_to_hub_controlnet = ComponentOp.from_registry(
    name="write_to_hf_hub",
    component_spec_path="components/write_to_hub_controlnet/fondant_component.yaml",
    arguments={
        "username": "test-user",
        "dataset_name": "segmentation_kfp",
        "hf_token": "hf_token",
        "image_column_names": ["images_data"],
    },
)

pipeline = Pipeline(pipeline_name=pipeline_name, base_path=PipelineConfigs.BASE_PATH)

pipeline.add_op(generate_prompts_op)
pipeline.add_op(laion_retrieval_op, dependencies=generate_prompts_op)
pipeline.add_op(download_images_op, dependencies=laion_retrieval_op)
pipeline.add_op(caption_images_op, dependencies=download_images_op)
pipeline.add_op(segment_images_op, dependencies=caption_images_op)
pipeline.add_op(write_to_hub_controlnet, dependencies=segment_images_op)
