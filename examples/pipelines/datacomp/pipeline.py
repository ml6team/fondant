"""Pipeline used to filter the dataset of the Datacomp competition."""

import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.compiler import DockerCompiler
from fondant.pipeline import ComponentOp, Pipeline, Client
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name="Datacomp filtering pipeline",
    pipeline_description="A pipeline for filtering the Datacomp dataset",
    base_path=PipelineConfigs.BASE_PATH,
)
client = Client(host=PipelineConfigs.HOST)

# define ops
load_component_column_mapping = {
    "url": "image_url",
    "original_width": "image_original_width",
    "original_height": "image_original_height",
    "face_bboxes": "image_face_bboxes",
    "sha256": "image_sha256",
    "text": "text_data",
    "clip_b32_similarity_score": "image_text_clip_b32_similarity_score",
    "clip_l14_similarity_score": "image_text_clip_b32_similarity_score",
}

load_from_hub_op = ComponentOp.from_registry(
    name="load_from_hub",
    component_spec_path="components/load_from_hf_hub/fondant_component.yaml",
    arguments={
        "dataset_name": "ml6team/the-stack-smol-python",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 100,
    },
)

# add ops to pipeline
pipeline.add_op(load_from_hub_op)
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
