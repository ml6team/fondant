"""Pipeline used to create a stable diffusion dataset from a set of given images."""
# pylint: disable=import-error
import json

from kfp import components as comp
from kfp import dsl

from config.general_config import KubeflowConfig
from fondant.pipeline_utils import compile_and_upload_pipeline

# Define metadata
base_path = KubeflowConfig.BASE_PATH
run_id = "{{workflow.name}}"
metadata = json.dumps({"base_path": base_path, "run_id": run_id})

# Define component ops
generate_prompts_op = comp.load_component(
    "components/generate_prompts/kubeflow_component.yaml"
)


# Pipeline
@dsl.pipeline(
    name="controlnet-pipeline",
    description="Pipeline that collects data to train ControlNet",
)
# pylint: disable=too-many-arguments, too-many-locals
def controlnet_pipeline(metadata: str = metadata):
    # Component 1
    generate_prompts_task = generate_prompts_op(metadata=metadata).set_display_name(
        "Generate initial prompts"
    )


if __name__ == "__main__":
    compile_and_upload_pipeline(
        pipeline=controlnet_pipeline,
        host=KubeflowConfig.HOST,
        env=KubeflowConfig.ENV,
        pipeline_id="80af809a-473b-4a4b-aa12-8d4d1d1cf882",
    )
