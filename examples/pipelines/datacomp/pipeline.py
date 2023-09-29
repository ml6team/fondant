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
        "dataset_uri": "gs://soy-audio-379412_datacomp/final_dataset_multiple/test2/",
        "column_name_mapping": load_component_column_mapping,
        "index_column": "uid",
        # "n_rows_to_load": 5000,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    memory_request="250G",
    cache=False,
)

detect_text_op = ComponentOp(
    component_dir="components/detect_text",
    arguments={
        "batch_size": 128,
    },
    node_pool_label="node_pool",
    node_pool_name="model-inference-pool",
    memory_request="250G",
    number_of_gpus=1,
    cache=False,
)
mask_images_op = ComponentOp(
    component_dir="components/mask_images",
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    memory_request="250G",
    cache=False,
)
embed_images_op = ComponentOp.from_registry(
    name="embed_images",
    arguments={
        "batch_size": 128,
    },
    node_pool_label="node_pool",
    node_pool_name="model-inference-pool",
    number_of_gpus=1,
    cache=False,
)
add_clip_score_op = ComponentOp(
    component_dir="components/add_clip_score",
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    memory_request="250G",
    cache=False,
)
filter_clip_score_op = ComponentOp(
    component_dir="components/filter_clip_score",
    arguments={
        "pct_threshold": 0.5,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-64-pool",
    cache=False,
)


# add ops to pipeline
pipeline.add_op(load_from_parquet)
pipeline.add_op(detect_text_op, dependencies=load_from_parquet)
pipeline.add_op(mask_images_op, dependencies=detect_text_op)
pipeline.add_op(embed_images_op, dependencies=mask_images_op)
pipeline.add_op(add_clip_score_op, dependencies=embed_images_op)
pipeline.add_op(filter_clip_score_op, dependencies=add_clip_score_op)
