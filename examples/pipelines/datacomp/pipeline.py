"""Pipeline used to filter the dataset of the Datacomp competition.

This pipeline implements the T-MARS paper: https://arxiv.org/abs/2307.03132.
"""
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

run_id = 1

# Global variable
IMAGE_SIZE = 256

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name=f"datacomp-filtering-pipeline-{run_id}-v2",
    pipeline_description="A pipeline for filtering the Datacomp dataset",
    base_path=PipelineConfigs.BASE_PATH,
)

# define ops
load_component_column_mapping = {
    "url": "images_url",
    "data": "images_data",
    "original_width": "images_width",
    "original_height": "images_height",
    "face_bboxes": "images_face_bboxes",
    "sha256": "images_sha256",
    "text": "text_data",
    "clip_b32_similarity_score": "imagetext_clipb32score",
    "clip_l14_similarity_score": "imagetext_clipl14score",
    "clip_l14_text_embedding": "textembedding_data",
}

load_from_parquet = ComponentOp(
    component_dir="components/load_from_parquet",
    arguments={
        "dataset_uri": f"gs://soy-audio-379412_datacomp/final_dataset_multiple/{run_id}/",
        "column_name_mapping": load_component_column_mapping,
        "index_column": "uid",
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
    memory_request="500G",
)

resize_images = ComponentOp.from_registry(
    name="resize_images",
    arguments={
        "resize_width": IMAGE_SIZE,
        "resize_height": IMAGE_SIZE,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
    memory_request="500G",
)

detect_text_op = ComponentOp(
    component_dir="components/detect_text",
    arguments={
        "batch_size": 8,
        "image_size": IMAGE_SIZE,
    },
    node_pool_label="node_pool",
    node_pool_name="model-inference-pool",
    accelerator_name="GPU",
    number_of_accelerators=1,
)
mask_images_op = ComponentOp(
    component_dir="components/mask_images",
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
    memory_request="500G",
)
embed_images_op = ComponentOp.from_registry(
    name="embed_images",
    arguments={
        "batch_size": 8,
    },
    node_pool_label="node_pool",
    node_pool_name="model-inference-pool",
    accelerator_name="GPU",
    number_of_accelerators=1,
)
add_clip_score_op = ComponentOp(
    component_dir="components/add_clip_score",
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
    memory_request="500G",
)
filter_clip_score_op = ComponentOp(
    component_dir="components/filter_clip_score",
    arguments={
        "threshold_score": 0.19,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
    memory_request="500G",
)

# add ops to pipeline
pipeline.add_op(load_from_parquet)
pipeline.add_op(resize_images, dependencies=load_from_parquet)
pipeline.add_op(detect_text_op, dependencies=resize_images)
pipeline.add_op(mask_images_op, dependencies=detect_text_op)
pipeline.add_op(embed_images_op, dependencies=mask_images_op)
pipeline.add_op(add_clip_score_op, dependencies=embed_images_op)
pipeline.add_op(filter_clip_score_op, dependencies=add_clip_score_op)
