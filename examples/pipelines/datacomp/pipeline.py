"""Pipeline used to filter the dataset of the Datacomp competition."""

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
    "clip_b32_similarity_score": "image_text_clip_b32_similarity_score",
    "clip_l14_similarity_score": "image_text_clip_l14_similarity_score",
}

load_from_hub_op = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "mlfoundations/datacomp_small",
        "column_name_mapping": load_component_column_mapping,
        "index_column": "uid",
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
)
download_images_op = ComponentOp.from_registry(
    name="download_images",
    arguments={
        "retries": 2,
        "min_image_size": 0,
        "max_aspect_ratio": float("inf"),
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
)
filter_complexity_op = ComponentOp(
    component_dir="components/filter_text_complexity",
    arguments={
        "spacy_pipeline": "en_core_web_sm",
        "batch_size": 1000,
        "min_complexity": 1,
    },
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
)


# add ops to pipeline
pipeline.add_op(load_from_hub_op)
# pipeline.add_op(filter_complexity_op, dependencies=download_images_op)
pipeline.add_op(download_images_op, dependencies=load_from_hub_op)

# TODO add more ops
