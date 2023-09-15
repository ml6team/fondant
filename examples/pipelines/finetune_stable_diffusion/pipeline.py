"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)
# General configs
pipeline_name = "stable_diffusion_pipeline"
pipeline_description = (
    "Pipeline to prepare and collect data for finetuning stable diffusion"
)

load_component_column_mapping = {"image": "images_data", "text": "captions_data"}

write_component_column_mapping = {
    value: key for key, value in load_component_column_mapping.items()
}
# Define component ops
load_from_hub_op = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "logo-wizard/modern-logo-dataset",
        "column_name_mapping": load_component_column_mapping,
        "image_column_names": ["image"],
        "n_rows_to_load": None,
    },
)

# Define component ops
image_resolution_extraction_op = ComponentOp.from_registry(
    name="image_resolution_extraction"
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
    number_of_accelerators=1,
    accelerator_name="GPU",
)

write_to_hub = ComponentOp(
    component_dir="components/write_to_hf_hub",
    arguments={
        "username": "test-user",
        "dataset_name": "stable_diffusion_processed",
        "hf_token": "hf_token",
        "image_column_names": ["images_data"],
    },
    number_of_accelerators=1,
    accelerator_name="GPU",
)

pipeline = Pipeline(
    pipeline_name=pipeline_name,
    base_path="/home/philippe/Scripts/express/local_artifact/new",
)

pipeline.add_op(load_from_hub_op)
# pipeline.add_op(image_resolution_extraction_op, dependencies=load_from_hub_op)
# pipeline.add_op(image_embedding_op, dependencies=image_resolution_extraction_op)
# pipeline.add_op(laion_retrieval_op, dependencies=image_embedding_op)
# pipeline.add_op(download_images_op, dependencies=laion_retrieval_op)
# pipeline.add_op(caption_images_op, dependencies=download_images_op)
# pipeline.add_op(write_to_hub, dependencies=caption_images_op)
