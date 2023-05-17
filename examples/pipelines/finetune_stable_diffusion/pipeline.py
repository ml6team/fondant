"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline, Client
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
# General configs
pipeline_name = "Test fondant pipeline"
pipeline_description = "A test pipeline"

client = Client(host=PipelineConfigs.HOST)

# Define component ops
load_from_hub_op = ComponentOp(
    component_spec_path="components/load_from_hub/fondant_component.yaml",
    arguments={"dataset_name": "logo-wizard/modern-logo-dataset"},
)

image_embedding_op = ComponentOp(
    component_spec_path="components/image_embedding/fondant_component.yaml",
    arguments={
        "min_width": 600,
        "min_height": 600,
    },
)

image_filtering_op = ComponentOp(
    component_spec_path="components/image_filtering/fondant_component.yaml",
    arguments={
        "min_width": 600,
        "min_height": 600,
    },
)

write_to_hub_op = ComponentOp(
    component_spec_path="components/write_to_hub/fondant_component.yaml",
    arguments={
        "username": "philippemo",
        "dataset_name": "test",
        "hf_token": "",
    },
)

pipeline = Pipeline(pipeline_name=pipeline_name, base_path=PipelineConfigs.BASE_PATH)

pipeline.add_op(load_from_hub_op)
pipeline.add_op(image_filtering_op, dependencies=load_from_hub_op)
#pipeline.add_op(image_filtering_op, dependencies=load_from_hub_op)
pipeline.add_op(write_to_hub_op, dependencies=image_filtering_op)

client.compile_and_run(pipeline=pipeline)
