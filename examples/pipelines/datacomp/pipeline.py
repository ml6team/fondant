"""Pipeline used to filter the dataset of the Datacomp competition.

This pipeline implements the T-MARS paper: https://arxiv.org/abs/2307.03132.
"""
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

# Global variable
IMAGE_SIZE = 256

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name="datacomp-filtering-pipeline",
    pipeline_description="A pipeline for filtering the Datacomp dataset",
    base_path=PipelineConfigs.BASE_PATH,
)

# define ops
load_component_column_mapping = {
    "url": "images_url",
    "original_width": "images_width",
    "original_height": "images_height",
    "face_bboxes": "images_face_bboxes",
    "sha256": "images_sha256",
    "text": "text_data",
    "uid": "image_text_uid",
    "clip_b32_similarity_score": "image_text_clip_b32_similarity_score",
    "clip_l14_similarity_score": "image_text_clip_l14_similarity_score",
}

load_from_hub_op = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "nielsr/datacomp-small-with-text-embeddings",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 1000,
    },
)

download_images_op = ComponentOp(
    component_dir="components/download_images",
    arguments={
        "retries": 2,
        "min_image_size": 0,
    },
)

resize_images = ComponentOp.from_registry(
    name="resize_images",
    arguments={
        "resize_width": IMAGE_SIZE,
        "resize_height": IMAGE_SIZE,
    },
)

detect_text_op = ComponentOp(
    component_dir="components/detect_text",
    arguments={
        "batch_size": 8,
        "image_size": IMAGE_SIZE,
    },
    accelerator_name="GPU",
    number_of_accelerators=1,
)
mask_images_op = ComponentOp(
    component_dir="components/mask_images",
)

embed_images_op = ComponentOp.from_registry(
    name="embed_images",
    arguments={
        "batch_size": 8,
    },
    accelerator_name="GPU",
    number_of_accelerators=1,
)
add_clip_score_op = ComponentOp(
    component_dir="components/add_clip_score",
)

filter_clip_score_op = ComponentOp(
    component_dir="components/filter_clip_score",
    arguments={
        "threshold_score": 0.19,
    },
)

# add ops to pipeline
pipeline.add_op(load_from_hub_op)
pipeline.add_op(download_images_op, dependencies=load_from_hub_op)
pipeline.add_op(resize_images, dependencies=download_images_op)
pipeline.add_op(detect_text_op, dependencies=resize_images)
pipeline.add_op(mask_images_op, dependencies=detect_text_op)
pipeline.add_op(embed_images_op, dependencies=mask_images_op)
pipeline.add_op(add_clip_score_op, dependencies=embed_images_op)
pipeline.add_op(filter_clip_score_op, dependencies=add_clip_score_op)
