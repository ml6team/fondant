"""
This module defines a pipeline with 2 Express components, a loading and a transform component.
"""

import json

from kfp import components as comp
from kfp import dsl

from config.general_config import GeneralConfig, KubeflowConfig

from express.pipeline_utils import compile_and_upload_pipeline

# Load Components
run_id = '{{workflow.name}}'
artifact_bucket = KubeflowConfig.ARTIFACT_BUCKET

# Component 1: load from hub
load_from_hub_op = comp.load_component('components/load_from_hub/component.yaml')
load_from_hub_extra_args = {"dataset_name": GeneralConfig.DATASET_NAME}
load_from_hub_metadata_args = {"run_id": run_id, "component_name": load_from_hub_op.__name__,
                               "artifact_bucket": artifact_bucket}
load_from_hub_extra_args = json.dumps(load_from_hub_extra_args)
load_from_hub_metadata_args = json.dumps(load_from_hub_metadata_args)

# Component 2: add captions
add_captions_op = comp.load_component('components/add_captions/component.yaml')
add_captions_extra_args = {}
add_captions_metadata_args = {"run_id": run_id, "component_name": add_captions_op.__name__,
                              "artifact_bucket": artifact_bucket}
add_captions_extra_args = json.dumps(add_captions_extra_args)
add_captions_metadata_args = json.dumps(add_captions_metadata_args)

a = {"t":1}
# Pipeline
@dsl.pipeline(
    name='HF Dataset pipeline',
    description='Tiny pipeline that includes 2 components to load and process a HF dataset'
)
def hf_dataset_pipeline(load_from_hub_extra_args: str = load_from_hub_extra_args,
                        load_from_hub_metadata_args: str = load_from_hub_metadata_args,
                        add_captions_extra_args: str = add_captions_extra_args,
                        add_captions_metadata_args: str = add_captions_metadata_args):
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

from typing import List, Dict
from kfp import dsl


def create_pipeline(components: List[Dict], pipeline_name: str):
    @dsl.pipeline(
        name=pipeline_name,
        description='Tiny pipeline that includes 2 components to load and process a HF dataset'
    )
    def pipeline():
        component_tasks = {}
        for component in components:
            component_name = component['name']
            component_op = component['op']
            component_inputs = component.get('inputs', {})
            component_outputs = component.get('outputs', {})
            component_display_name = component.get('display_name', component_name)

            # Create task for component
            print(component_display_name)
            task = component_op(**component_inputs)
            if component_outputs:
                for output_name, output in component_outputs.items():
                    component_tasks[output_name] = task.outputs[output]

        return component_tasks

    return pipeline


components = [{'name': 'load_from_hub', 'op': load_from_hub_op,
               'inputs': {'extra_args': load_from_hub_extra_args,
                          'metadata_args': load_from_hub_metadata_args},
               'outputs': {'output_manifest': 'output_manifest'},
               'display_name': 'Load from Hub Component', },
              {'name': 'add_captions', 'op': add_captions_op,
               'inputs': {'input_manifest': "11",
                          'extra_args': add_captions_extra_args,
                          'metadata': add_captions_metadata_args},
               'outputs': {'output_manifest': 'output_manifest'},
               'display_name': 'Add Captions Component', }, ]

# pipeline_func = create_pipeline(components=components, pipeline_name='my_pipeline')
# pipeline_func()
