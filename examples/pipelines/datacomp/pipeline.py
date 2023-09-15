"""Pipeline used to filter the dataset of the Datacomp competition.

This pipeline implements the T-MARS paper: https://arxiv.org/abs/2307.03132.
"""

import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name="datacomp-filtering-pipeline",
    pipeline_description="A pipeline for filtering the Datacomp dataset",
    base_path=PipelineConfigs.BASE_PATH,
    # base_path="/Users/nielsrogge/Documents/fondant_artifacts_datacomp",
)

# define ops
load_component_column_mapping = {
    "url": "images_url",
    "original_width": "images_width",
    "original_height": "images_height",
    "face_bboxes": "images_face_bboxes",
    "sha256": "images_sha256",
    "text": "text_data",
    "clip_b32_similarity_score": "imagetext_clipb32score",
    "clip_l14_similarity_score": "imagetext_clipl14score",
    "clip_l14_text_embedding": "textembedding_data",
}

load_from_hub_op = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "nielsr/datacomp-small-with-text-embeddings",
        "column_name_mapping": load_component_column_mapping,
        "index_column": "uid",
        "n_rows_to_load": 1000,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    cache=False,
)
download_images_op = ComponentOp.from_registry(
    name="download_images",
    arguments={
        "retries": 2,
        "min_image_size": 0,
        "max_aspect_ratio": float("inf"),
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    input_partition_rows=1000,
    cache=False,
)
detect_text_op = ComponentOp(
    component_dir="components/detect_text",
    arguments={
        "batch_size": 2,
    },
    node_pool_label="node_pool",
    node_pool_name="model-inference-mega-pool",
    number_of_accelerators=1,
    accelerator_name="GPU",
    cache=False,
)
mask_images_op = ComponentOp(
    component_dir="components/mask_images",
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    cache=False,
)
embed_images_op = ComponentOp.from_registry(
    name="image_embedding",
    arguments={
        "batch_size": 2,
    },
    node_pool_label="node_pool",
    node_pool_name="model-inference-mega-pool",
    number_of_accelerators=1,
    accelerator_name="GPU",
    cache=False,
)
add_clip_score_op = ComponentOp(
    component_dir="components/add_clip_score",
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    cache=False,
)
filter_clip_score_op = ComponentOp(
    component_dir="components/filter_clip_score",
    arguments={
        "pct_threshold": 0.5,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
)

# add ops to pipeline
pipeline.add_op(load_from_hub_op)
# pipeline.add_op(download_images_op, dependencies=load_from_hub_op)
# pipeline.add_op(detect_text_op, dependencies=download_images_op)
# pipeline.add_op(mask_images_op, dependencies=detect_text_op)
# pipeline.add_op(embed_images_op, dependencies=mask_images_op)
# pipeline.add_op(add_clip_score_op, dependencies=embed_images_op)
# pipeline.add_op(filter_clip_score_op, dependencies=add_clip_score_op)
