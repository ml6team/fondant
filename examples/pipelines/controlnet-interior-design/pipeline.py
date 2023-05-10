"""Pipeline used to create a stable diffusion dataset from a set of initial prompts."""
# pylint: disable=import-error
import json

from kfp import components as comp
from kfp import dsl

from config.general_config import KubeflowConfig
from config.components_config import RetrieveImagesConfig
from fondant.pipeline_utils import compile_and_upload_pipeline

# Define metadata
base_path = KubeflowConfig.BASE_PATH
run_id = "{{workflow.name}}"
metadata = json.dumps({"base_path": base_path, "run_id": run_id})

# Define component ops
generate_prompts_op = comp.load_component(
    "components/generate_prompts/kubeflow_component.yaml"
)
retrieve_images_op = comp.load_component(
    "components/retrieve_images/kubeflow_component.yaml"
)


# Pipeline
@dsl.pipeline(
    name="controlnet-pipeline",
    description="Pipeline that collects data to train ControlNet",
)
# pylint: disable=too-many-arguments, too-many-locals
def controlnet_pipeline(
    metadata: str = metadata,
    retrieve_images_num_images: int = RetrieveImagesConfig.NUM_IMAGES,
    retrieve_images_aesthetic_score: int = RetrieveImagesConfig.AESTHETIC_SCORE,
    retrieve_images_aesthetic_weight: float = RetrieveImagesConfig.AESTHETIC_WEIGHT,
):
    # Component 1
    generate_prompts_task = generate_prompts_op(metadata=metadata).set_display_name(
        "Generate initial prompts"
    )

    # Component 2
    retrieve_images_task = retrieve_images_op(
        input_manifest_path=generate_prompts_task.outputs["output_manifest_path"],
        num_images=retrieve_images_num_images,
        aesthetic_score=retrieve_images_aesthetic_score,
        aesthetic_weight=retrieve_images_aesthetic_weight,
    ).set_display_name("Retrieve image URLs")


if __name__ == "__main__":
    compile_and_upload_pipeline(
        pipeline=controlnet_pipeline,
        host=KubeflowConfig.HOST,
        env=KubeflowConfig.ENV,
        pipeline_id="176770b4-0c12-4e57-8c2a-5b9c2af56ec6",
    )
