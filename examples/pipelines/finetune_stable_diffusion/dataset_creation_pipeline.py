"""Pipeline used to create a stable diffusion dataset from a set of given images. This is done by
using clip retrieval on the LAION dataset"""
# pylint: disable=import-error
import json

from kfp import components as comp
from kfp import dsl

from config.general_config import GeneralConfig, KubeflowConfig
from config.components_config import DatasetLoaderConfig, ImageFilterConfig
from express.kfp_utils import compile_and_upload_pipeline

# Load Components
run_id = '{{workflow.name}}'
artifact_bucket = KubeflowConfig.ARTIFACT_BUCKET

dataset_loader_op = comp.load_component('components/dataset_loader_component/component.yaml')
dataset_loader_extra_args = {"project_id": GeneralConfig.GCP_PROJECT_ID, "bucket": DatasetLoaderConfig.BUCKET_NAME}
dataset_loader_metadata_args = {"run_id": run_id, "component_name": dataset_loader_op.__name__, "artifact_bucket": artifact_bucket}
dataset_loader_extra_args = json.dumps(dataset_loader_extra_args)
dataset_loader_metadata_args = json.dumps(dataset_loader_metadata_args)


# Pipeline
@dsl.pipeline(
    name='image-generator-dataset',
    description='Pipeline that takes example images as input and returns an expanded dataset of '
                'similar images as outputs'
)
# pylint: disable=too-many-arguments, too-many-locals
def sd_dataset_creator_pipeline(dataset_loader_extra_args: str = dataset_loader_extra_args,
                        dataset_loader_metadata_args: str = dataset_loader_metadata_args,
):
    # Component 1
    dataset_loader_task = dataset_loader_op(extra_args=dataset_loader_extra_args,
                                          metadata_args=dataset_loader_metadata_args,
    ).set_display_name('Load initial images')


if __name__ == '__main__':
    compile_and_upload_pipeline(pipeline=sd_dataset_creator_pipeline,
                                host=KubeflowConfig.HOST,
                                env=KubeflowConfig.ENV)
