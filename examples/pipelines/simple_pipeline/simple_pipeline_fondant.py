"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging
import sys

sys.path.append('../')

from pipeline_configs import PipelineConfigs

from fondant.pipeline import FondantComponentOp, FondantPipeline, FondantClient
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
# General configs
pipeline_name = "Test fondant pipeline"
pipeline_description = "A test pipeline"
pipeline_host = PipelineConfigs.HOST
pipeline_base_path = PipelineConfigs.BASE_PATH

client = FondantClient(host=pipeline_host)

# Define component ops
load_from_hub_op = FondantComponentOp(
    component_spec_path="components/load_from_hub/fondant_component.yaml",
    arguments={"dataset_name": "lambdalabs/pokemon-blip-captions"},
)

image_filtering_op = FondantComponentOp(
    component_spec_path="components/image_filtering/fondant_component.yaml",
    arguments={
        "min_width": 600,
        "min_height": 600,
    },
)

# TODO: ADD Arguments for embedding component later on
# MODEL_ID = "openai/clip-vit-large-patch14"
# BATCH_SIZE = 10

pipeline = FondantPipeline(pipeline_name=pipeline_name, base_path=pipeline_base_path)

pipeline.add_op(load_from_hub_op)
pipeline.add_op(image_filtering_op, dependencies=load_from_hub_op)

client.compile_and_run(pipeline=pipeline)
