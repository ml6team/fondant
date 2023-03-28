"""
This module defines a pipeline with 2 Express components, a loading and a transform component.
"""

from kfp import components as comp
from kfp import dsl

from kubernetes import client as k8s_client

from config.pipeline_config import KubeflowConfig, LoadFromHubConfig

from express.pipeline_utils import create_extra_args, create_metadata_args, compile_and_upload_pipeline

# Load Components
artifact_bucket = KubeflowConfig.ARTIFACT_BUCKET

# Component 1: load from hub
load_from_hub_component = comp.load_component('components/load_from_hub/component.yaml')
load_from_hub_extra_args = create_extra_args(dataset_name=LoadFromHubConfig.DATASET_NAME)
load_from_hub_metadata_args = create_metadata_args(load_from_hub_component, artifact_bucket)

# Component 2: add captions
add_captions_component = comp.load_component('components/add_captions/component.yaml')
add_captions_extra_args = create_extra_args()
add_captions_metadata_args = create_metadata_args(add_captions_component, artifact_bucket)

# Pipeline
@dsl.pipeline(
    name='HF Dataset pipeline',
    description='Tiny pipeline that includes 2 components to load and process a HF dataset'
)
def hf_dataset_pipeline(load_from_hub_extra_args: str = load_from_hub_extra_args,
                        load_from_hub_metadata_args: str = load_from_hub_metadata_args,
                        add_captions_extra_args: str = add_captions_extra_args,
                        add_captions_metadata_args: str = add_captions_metadata_args,
                        ):
    # Component 1
    load_from_hub_task = load_from_hub_component(extra_args=load_from_hub_extra_args,
                                          metadata_args=load_from_hub_metadata_args,
    ).set_display_name('Load from hub component')

    # Component 2
    add_captions_task = add_captions_component(extra_args=add_captions_extra_args,
                                        metadata=add_captions_metadata_args,
                                        input_manifest=load_from_hub_task.outputs["output_manifest"],
    ).set_display_name('Add captions component') \
    .set_gpu_limit(1) \
    .add_node_selector_constraint('node_pool', 'model-inference-pool') \
    .add_toleration(
        k8s_client.V1Toleration(effect='NoSchedule', key='reserved-pool', operator='Equal',
                                value='true'))


if __name__ == '__main__':
    compile_and_upload_pipeline(pipeline=hf_dataset_pipeline,
                                host=KubeflowConfig.HOST,
                                env=KubeflowConfig.ENV)