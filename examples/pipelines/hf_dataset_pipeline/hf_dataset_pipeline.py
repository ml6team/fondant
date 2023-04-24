"""
This module defines a pipeline with 2 Fondant components, a loading and a transform component.
"""

import json

from kfp import components as comp
from kfp import dsl

from config.general_config import GeneralConfig, KubeflowConfig

from fondant.kfp_utils import compile_and_upload_pipeline

# Load Components
run_id = '{{workflow.name}}'
artifact_bucket = KubeflowConfig.ARTIFACT_BUCKET

# Component 1: load from hub
load_from_hub_op = comp.load_component('components/load_from_hub/component.yaml')
load_from_hub_extra_args = {"dataset_name": GeneralConfig.DATASET_NAME}
load_from_hub_metadata_args = {"run_id": run_id, "component_name": load_from_hub_op.__name__, "artifact_bucket": artifact_bucket}
load_from_hub_extra_args = json.dumps(load_from_hub_extra_args)
load_from_hub_metadata_args = json.dumps(load_from_hub_metadata_args)

# Component 2: add captions
add_captions_op = comp.load_component('components/add_captions/component.yaml')
add_captions_extra_args = {}
add_captions_metadata_args = {"run_id": run_id, "component_name": add_captions_op.__name__, "artifact_bucket": artifact_bucket}
add_captions_extra_args = json.dumps(add_captions_extra_args)
add_captions_metadata_args = json.dumps(add_captions_metadata_args)

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
    load_from_hub_task = load_from_hub_op(extra_args=load_from_hub_extra_args,
                                          metadata_args=load_from_hub_metadata_args,
    ).set_display_name('Load from hub component')

    # Component 2
    add_captions_op(extra_args=add_captions_extra_args,
                                        metadata=add_captions_metadata_args,
                                        input_manifest=load_from_hub_task.outputs["output_manifest"],
    ).set_display_name('Add captions component')


if __name__ == '__main__':
    compile_and_upload_pipeline(pipeline=hf_dataset_pipeline,
                                host=KubeflowConfig.HOST,
                                env=KubeflowConfig.ENV)