"""Pipeline used to filter the dataset of the Datacomp competition."""

import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.compiler import DockerCompiler
from fondant.pipeline import ComponentOp, Pipeline, Client

logger = logging.getLogger(__name__)

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name="Datacomp filtering pipeline",
    pipeline_description="A pipeline for filtering the Datacomp dataset",
    # base_path=PipelineConfigs.BASE_PATH,
    base_path="/Users/nielsrogge/Documents/fondant_artifacts_datacomp",
)
client = Client(host=PipelineConfigs.HOST)

# define ops
load_component_column_mapping = {
    "url": "image_url",
    "original_width": "image_width",
    "original_height": "image_height",
    "face_bboxes": "image_face_bboxes",
    "sha256": "image_sha256",
    "clip_l14_embedding": "image_embedding",
    "text": "text_data",
    "clip_b32_similarity_score": "image_text_clip_b32_similarity_score",
    "clip_l14_similarity_score": "image_text_clip_l14_similarity_score",
}

load_from_hub_op = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "nielsr/datacomp-small-with-embeddings",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 100,
    },
)
filter_image_resolution_op = ComponentOp.from_registry(
    name="filter_image_resolution",
    arguments={"min_image_dim": 200, "max_aspect_ratio": 3},
)
filter_complexity_op = ComponentOp(
    component_dir="components/filter_text_complexity",
    arguments={
        "spacy_pipeline": "en_core_web_sm",
        "batch_size": 1000,
        "min_complexity": 1,
        "min_num_actions": 1,
    },
)
cluster_image_embeddings_op = ComponentOp(
    component_dir="components/cluster_image_embeddings",
    arguments={
        "sample_ratio": 0.3,
        "num_clusters": 3,
    },
)
dedup_image_embeddings_op = ComponentOp.from_registry(
    name="dedup_image_embeddings",
    component_spec_path="components/dedup_image_embeddings/fondant_component.yaml",
    arguments={
        "epsilon": 0.03,
    },
)

# add ops to pipeline
pipeline.add_op(load_from_hub_op)
pipeline.add_op(filter_image_resolution_op, dependencies=load_from_hub_op)
pipeline.add_op(filter_complexity_op, dependencies=filter_image_resolution_op)
pipeline.add_op(cluster_image_embeddings_op, dependencies=filter_complexity_op)
pipeline.add_op(dedup_image_embeddings_op, dependencies=cluster_image_embeddings_op)
# TODO add more ops

# compile
if __name__ == "__main__":
    compiler = DockerCompiler()
    # mount the gcloud credentials to the container
    extra_volumes = [
        "$HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json:ro"
    ]
    compiler.compile(pipeline=pipeline, extra_volumes=extra_volumes)
    logger.info("Run `docker compose up` to run the pipeline.")
