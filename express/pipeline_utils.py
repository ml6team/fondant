"""General pipeline utils"""

import os
import logging
import yaml
from typing import Callable, List

from express.import_utils import is_kfp_available

if is_kfp_available():
    import kfp

logger = logging.getLogger(__name__)


def find_dict_by_attribute(
    list_of_dicts: List[dict], attribute_name: str, attribute_value: any
) -> dict:
    """
    Function that searches a dict from a list of dicts based on an attribute value
    Args:
        list_of_dicts: the list of dicts to search
        attribute_name: the attribute name
        attribute_value: the attribute value
    Returns:
        the dictionary with the desired attribute
    """
    return next(
        (d for d in list_of_dicts if d.get(attribute_name) == attribute_value), None
    )


def find_dict_without_attribute(list_of_dicts, attribute_name) -> dict:
    """
    Function that searches a dict from a list of dicts based on an missing attribute value
    Args:
        list_of_dicts: the list of dicts to search
        attribute_name: the attribute name
    Returns:
        the dictionary with the missing attribute
    """
    return next((d for d in list_of_dicts if attribute_name not in d), None)


def get_components_order(pipeline_name: str, pipeline_dict: dict) -> List[str]:
    """
    Function that returns a list of the pipeline component in the defined order of execution
    Args:
        pipeline_name: the name of the pipeline
        pipeline_dict: the pipeline dictionary of the yaml file
    Returns:
        list of component names in the order of execution
    """
    component_list = []
    tasks = pipeline_dict["spec"]["templates"]
    pipeline_specs = find_dict_by_attribute(tasks, "name", pipeline_name)
    pipeline_tasks = pipeline_specs["dag"]["tasks"]
    current_task = find_dict_without_attribute(pipeline_tasks, "dependencies")
    current_component_name = current_task["name"]
    component_list.append(current_component_name)
    pipeline_tasks.remove(current_task)
    while True:
        for current_task in pipeline_tasks:
            if (
                "dependencies" in current_task
                and current_component_name in current_task["dependencies"]
            ):
                current_component_name = current_task["name"]
                component_list.append(current_component_name)
                pipeline_tasks.remove(current_task)
        else:
            break
    return component_list


def validate_subset_schema(pipeline_name: str, pipeline_file: str):
    """
    Function that validates the schema of the pipeline according to the subset specifications
    Args:
        pipeline_name: the name of the pipeline
        pipeline_file: the compiled pipeline file
    """

    def _get_annotations_dict(pipeline_dictionary, component_name):
        task_dict = pipeline_dictionary["spec"]["templates"]
        return find_dict_by_attribute(task_dict, "name", component_name)["metadata"][
            "annotations"
        ]

    available_subsets = []
    load_component = True
    with open(pipeline_file, "r") as stream:
        pipeline_dict = yaml.safe_load(stream)

        ordered_components = get_components_order(pipeline_name, pipeline_dict)
        for component in ordered_components:
            annotation_dict = _get_annotations_dict(pipeline_dict, component)
            available_subsets.append(annotation_dict["output_subsets"])
            if not load_component:
                if annotation_dict["input_subsets"] not in available_subsets:
                    logger.error(
                        "Component subset specifications are unmet. The component %s expects"
                        "the dataset to contain the %s subset but it was not created in any of "
                        "the previous components."
                        "Please make sure you have a component that created this subset and that"
                        " it's defined properly in the component specifications."
                    )
            load_component = False
    logger.warning("Component specifications match ")


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
