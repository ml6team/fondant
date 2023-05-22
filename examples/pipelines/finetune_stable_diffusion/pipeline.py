"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline, Client
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
# General configs
pipeline_name = "Test fondant pipeline"
pipeline_description = "A test pipeline"

client = Client(host=PipelineConfigs.HOST)

# Define component ops
load_from_hub_op = ComponentOp.from_registry(
    name="load_from_hf_hub",
    arguments={"dataset_name": "logo-wizard/modern-logo-dataset"},
)

image_embedding_op = ComponentOp.from_registry(
    name="image_embedding",
    arguments={
        "model_id": "openai/clip-vit-large-patch14",
        "batch_size": 10,
    },
)

laion_retrieval_op = ComponentOp.from_registry(
    name="embedding_based_laion_retrieval",
    arguments={"num_images": 2, "aesthetic_score": 9, "aesthetic_weight": 0.5},
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


pipeline = Pipeline(pipeline_name=pipeline_name, base_path=PipelineConfigs.BASE_PATH)

pipeline.add_op(load_from_hub_op)
pipeline.add_op(image_embedding_op, dependencies=load_from_hub_op)
pipeline.add_op(laion_retrieval_op, dependencies=image_embedding_op)
pipeline.add_op(download_images_op, dependencies=laion_retrieval_op)
pipeline.add_op(caption_images_op, dependencies=download_images_op)

client.compile_and_run(pipeline=pipeline)
