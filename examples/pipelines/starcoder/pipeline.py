"""Pipeline used to create the dataset to train the StarCoder model."""

import argparse
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.compiler import DockerCompiler
from fondant.logger import configure_logging
from fondant.pipeline import Client, ComponentOp, Pipeline

configure_logging()
logger = logging.getLogger(__name__)


client = Client(host=PipelineConfigs.HOST)

dataset_column_name = [
    "content",
    "lang",
    "size",
    "path",
    "repository_name",
    "avg_line_length",
    "max_line_length",
    "alphanum_fraction",
]

load_component_column_mapping = {
    column: f"code_{column}" for column in dataset_column_name
}

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name="Stack filtering pipeline",
    pipeline_description="A pipeline for filtering the stack dataset",
    base_path=PipelineConfigs.BASE_PATH,
)

# define ops
load_from_hub_op = ComponentOp.from_registry(
    name="load_from_hub",
    component_spec_path="components/load_from_hub/fondant_component.yaml",
    arguments={
        "dataset_name": "ml6team/the-stack-smol-python",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": None,
    },
)

filter_line_length_op = ComponentOp.from_registry(
    name="filter_line_length",
    arguments={
        "avg_line_length_threshold": 10,
        "max_line_length_threshold": 100,
        "alphanum_fraction_threshold": 0.25,
    },
)

filter_comments_op = ComponentOp.from_registry(
    name="filter_comments",
    arguments={"min_comments_ratio": 0.1, "max_comments_ratio": 0.9},
)
pii_redaction_op = ComponentOp.from_registry(
    name="pii_redaction",
)

# add ops to pipeline
pipeline.add_op(load_from_hub_op)
pipeline.add_op(filter_line_length_op, dependencies=load_from_hub_op)
pipeline.add_op(filter_comments_op, dependencies=filter_line_length_op)
pipeline.add_op(pii_redaction_op, dependencies=filter_comments_op)


if __name__ == "__main__":
    compiler = DockerCompiler()
    # mount the gcloud credentials to the container
    extra_volumes = [
        "$HOME/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json:ro"
    ]
    compiler.compile(pipeline=pipeline, extra_volumes=extra_volumes)
    logger.info("Run `docker compose up` to run the pipeline.")
