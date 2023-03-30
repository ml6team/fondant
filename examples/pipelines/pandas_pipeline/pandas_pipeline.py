"""
This module defines a pipeline with 2 Express components, a loading and a transform component.
"""

import json
import os
import sys

from kfp import components as comp
from kfp import dsl

sys.path.insert(0, os.path.abspath('..'))

from config import KubeflowConfig
from pipelines_config import LoadFromCloudConfig
from express.pipeline_utils import compile_and_upload_pipeline

# Load Components
run_id = '{{workflow.name}}'
artifact_bucket = KubeflowConfig.ARTIFACT_BUCKET

# Component 1: load from hub
load_from_cloud_op = comp.load_component('components/load_from_cloud/component.yaml')
load_from_cloud_extra_args = {"dataset_remote_path": LoadFromCloudConfig.DATASET_REMOTE_PATH}
load_from_cloud_metadata_args = {"run_id": run_id, "component_name": load_from_cloud_op.__name__,
                                 "artifact_bucket": artifact_bucket}
load_from_cloud_extra_args = json.dumps(load_from_cloud_extra_args)
load_from_cloud_metadata_args = json.dumps(load_from_cloud_metadata_args)

# Component 2: filter images
filter_images_op = comp.load_component('components/filter_images/component.yaml')
filter_images_extra_args = {"dataset_remote_path": LoadFromCloudConfig.DATASET_REMOTE_PATH}
filter_images_metadata_args = {"run_id": run_id, "component_name": load_from_cloud_op.__name__,
                               "artifact_bucket": artifact_bucket}
filter_images_extra_args = json.dumps(filter_images_extra_args)
filter_images_metadata_args = json.dumps(filter_images_metadata_args)


# Pipeline
@dsl.pipeline(
    name='Pandas Dataset pipeline',
    description='Tiny pipeline that includes 2 components to load and process a pandas dataset'
)
def pandas_dataset_pipeline(load_from_cloud_extra_args: str = load_from_cloud_extra_args,
                            load_from_cloud_metadata_args: str = load_from_cloud_metadata_args,
                            filter_images_extra_args: str = filter_images_extra_args,
                            filter_images_metadata_args: str = filter_images_metadata_args):
    # Component 1
    load_from_cloud_task = load_from_cloud_op(
        extra_args=load_from_cloud_extra_args,
        metadata_args=load_from_cloud_metadata_args,
    ).set_display_name('Load from cloud component')

    # Component 2
    filter_images_task = load_from_cloud_op(
        extra_args=filter_images_extra_args,
        metadata_args=filter_images_metadata_args,
        input_manifest=load_from_cloud_task.outputs["output_manifest"]
    ).set_display_name('Filter images component')


if __name__ == '__main__':
    compile_and_upload_pipeline(pipeline=pandas_dataset_pipeline,
                                host=KubeflowConfig.HOST,
                                env=KubeflowConfig.ENV)
