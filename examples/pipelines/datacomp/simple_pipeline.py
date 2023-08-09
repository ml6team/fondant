"""Simplified pipeline used to filter the dataset of the Datacomp competition."""

import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline, Client

logger = logging.getLogger(__name__)

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name="datacomp-filtering",
    pipeline_description="A pipeline for filtering the Datacomp dataset",
    base_path=PipelineConfigs.BASE_PATH,
)
client = Client(host=PipelineConfigs.HOST)

# define ops
load_component_column_mapping = {
    "url": "images_url",
    "original_width": "images_width",
    "original_height": "images_height",
    "face_bboxes": "images_face_bboxes",
    "sha256": "images_sha256",
    "text": "text_data",
    "clip_b32_similarity_score": "image_text_clip_b32_similarity_score",
    "clip_l14_similarity_score": "image_text_clip_l14_similarity_score",
}

load_from_hub_op = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "nielsr/datacomp-small-with-embeddings",
        "column_name_mapping": load_component_column_mapping,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
)
filter_image_resolution_op = ComponentOp.from_registry(
    name="filter_image_resolution",
    arguments={"min_image_dim": 200, "max_aspect_ratio": 3},
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
)
filter_complexity_op = ComponentOp(
    component_dir="components/filter_text_complexity",
    arguments={
        "spacy_pipeline": "en_core_web_sm",
        "batch_size": 1000,
        "min_complexity": 1,
        "min_num_actions": 0,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
    output_partition_size="disable",
)

# add ops to pipeline
pipeline.add_op(load_from_hub_op)
pipeline.add_op(filter_image_resolution_op, dependencies=load_from_hub_op)
pipeline.add_op(filter_complexity_op, dependencies=filter_image_resolution_op)
# TODO add more ops


if __name__ == "__main__":
    client.compile_and_run(pipeline=pipeline)
