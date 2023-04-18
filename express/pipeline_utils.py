"""General pipeline utils"""

import os
import logging
import json
from typing import Callable, Dict
from kfp import components as comp
from kfp import dsl
from express.import_utils import is_kfp_available
from express.component_spec import ExpressComponent

if is_kfp_available():
    import kfp

logger = logging.getLogger(__name__)


def compile_and_upload_pipeline(
        pipeline: Callable[[], None], host: str, env: str
) -> None:
    """Upload pipeline to kubeflow.
    Args:
        pipeline (Callable): function that contains the pipeline definition
        host (str): the url host for kfp
        env (str): the project run environment (sbx, dev, prd)
    """
    client = kfp.Client(host=host)

    pipeline_name = f"{pipeline.__name__}:{env}"
    pipeline_filename = f"{pipeline_name}.yaml"
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=pipeline_filename
    )

    existing_pipelines = client.list_pipelines(page_size=100).pipelines
    for existing_pipeline in existing_pipelines:
        if existing_pipeline.name == pipeline_name:
            # Delete existing pipeline before uploading
            logger.warning(
                f"Pipeline {pipeline_name} already exists. Deleting old pipeline..."
            )
            client.delete_pipeline_version(existing_pipeline.default_version.id)
            client.delete_pipeline(existing_pipeline.id)

    logger.info(f"Uploading pipeline: {pipeline_name}")
    client.upload_pipeline(pipeline_filename, pipeline_name=pipeline_name)
    os.remove(pipeline_filename)


def get_container_op(express_component: ExpressComponent) -> Callable:
    return comp.load_component(text=json.dumps(express_component.kubeflow_component_specification))


express_component_yaml_path = \
    "/home/philippe/Scripts/express/tests/component_example/valid_component/express_component.yaml"
component_op = get_container_op(ExpressComponent(express_component_yaml_path))
component_args = {"input_manifest_path": "aa", "args": "aa", "storage_args": "aa"}


@dsl.pipeline(name='pipeline name',
              description="""pipeline description""")
def pipeline(input_manifest_path:str="aa"):
    initial_op = component_op(input_manifest_path=input_manifest_path,
                              args=component_args["args"],
                              storage_args=component_args["storage_args"])


compile_and_upload_pipeline(pipeline,
                            "https://472c61c751ab9be9-dot-europe-west1.pipelines.googleusercontent.com",
                            "baaaaw")
# def create_pipeline(components: List[Dict], name: str, description: str = "test"):
#     @dsl.pipeline(
#         name=name,
#         description=description
#     )
#     def pipeline():
#         component_tasks = {}
#         for component in components:
#             component_name = component['name']
#             component_op = component['op']
#             component_inputs = component.get('inputs', {})
#             component_outputs = component.get('outputs', {})
#             component_display_name = component.get('display_name', component_name)
#
#             # Create task for component
#             print(component_display_name)
#             task = component_op(**component_inputs)
#             if component_outputs:
#                 for output_name, output in component_outputs.items():
#                     component_tasks[output_name] = task.outputs[output]
#
#         return component_tasks
#
#     return pipeline
#
#
# comp_1_args = {"storage_args": "test"}
# component_1 = ExpressComponent("express_component.yaml")
# component_2 = ExpressComponent("express_component2.yaml")
# component_2_args = {"storage_args": "aaa"}
# component_2_args = {"storage_args": "aaa"}
# component_list = [component_1, component_2]
#
# for component_idx, component in enumerate(component_list):
#     if component_idx == 0:
#         input_manifest_path = "None"
#     else:
#         input_manifest_path = "take the last one"
#
# component_inputs = {"input_manifest_path": "aa",
#                     "storage_args": "test",
#                     "args": "11"}
# component_output = {"outputs": "test"}
# pat = "/home/philippe/Scripts/express/tests/component_example/valid_component/express_component.yaml"
# component_1 = ExpressComponent(pat)
# # Create op
# component_op_1 = comp.load_component(
#     "/home/philippe/Scripts/express/tests/component_example/valid_component/kubeflow_component.yaml")
# task_1 = component_op_1(**component_inputs)
# a = 2
