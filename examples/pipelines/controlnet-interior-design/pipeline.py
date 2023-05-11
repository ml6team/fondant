"""Pipeline used to create a stable diffusion dataset from a set of initial prompts."""
# pylint: disable=import-error
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import FondantComponentOp, FondantPipeline, FondantClient
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
# General configs
pipeline_name = "controlnet-pipeline"
pipeline_description = "Pipeline that collects data to train ControlNet"

client = FondantClient(host=PipelineConfigs.HOST)

# Define component ops
generate_prompts_op = FondantComponentOp(
    component_spec_path="components/generate_prompts/fondant_component.yaml"
)

laion_retrieval_op = FondantComponentOp(
    component_spec_path="components/laion_retrieval/fondant_component.yaml",
    arguments={"num_images": 2, "aesthetic_score": 9, "aesthetic_weight": 0.5},
)

pipeline = FondantPipeline(
    pipeline_name=pipeline_name, base_path=PipelineConfigs.BASE_PATH
)

pipeline.add_op(generate_prompts_op)
pipeline.add_op(laion_retrieval_op, dependencies=generate_prompts_op)

client.compile_and_run(pipeline=pipeline)
