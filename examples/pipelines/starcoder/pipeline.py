"""Pipeline used to create the dataset to train the StarCoder model."""

import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline, Client
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Initialize pipeline and client
pipeline = Pipeline(
    pipeline_name="Stack filtering pipeline",
    pipeline_description="A pipeline for filtering the stack dataset",
    base_path=PipelineConfigs.BASE_PATH,
)
client = Client(host=PipelineConfigs.HOST)

# define ops
load_from_hub_op = ComponentOp(
    component_spec_path="components/load_from_hub_stack/fondant_component.yaml",
    arguments={"dataset_name": "ml6team/the-stack-smol-python"},
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
pipeline.add_op(pii_redaction_op, dependencies=load_from_hub_op)

# compile
client.compile_and_run(pipeline=pipeline)
