"""This module defines classes to represent a Fondant Pipeline."""
import logging
import yaml
import typing as t
from dataclasses import dataclass
from pathlib import Path

from fondant.component import FondantComponentSpec
from fondant.component_spec import ComponentSubset
from fondant.import_utils import is_kfp_available
from fondant.exceptions import InvalidPipelineDefinition

if is_kfp_available():
    import kfp.dsl as dsl
    import kfp
    from kubernetes import client as k8s_client

logger = logging.getLogger(__name__)


@dataclass
class FondantComponentOperation(FondantComponentSpec):
    """
    Class representing an operation for a Fondant Component in a Kubeflow Pipeline. An operation
    is a representation of a function that will be executed as part of a pipeline.

    Arguments:
        component_spec_path: The path to the specification file defining the component
        arguments: A dictionary containing the argument name and value for the operation.
        number_of_gpus: The number of gpus to assign to the operation
        node_pool_name: The name of the node pool to which the operation will be assigned.
        p_volumes: Collection of persistent volumes in a Kubernetes cluster. Keys are mount paths,
         values are Kubernetes volumes or inherited types(e.g. PipelineVolumes).
        ephemeral_storage_request = Used ephemeral-storage request (minimum) for the operation.
         Defined by string which can be a number or a number followed by one of “E”, “P”, “T”, “G”,
         “M”, “K”. (e.g. 2T for 2 Terabytes)

    Note:
        - A Fondant Component operation is created by defining a Fondant Component and its input
            arguments.
        - The `number_of_gpus`, `node_pool_name`,`p_volumes` and `ephemeral_storage_size`
            attributes are optional and can be used to specify additional configurations for
            the operation. More information on the optional attributes that can be assigned to
             kfp components here:
             https://kubeflow-pipelines.readthedocs.io/en/1.8.13/source/kfp.dsl.html
    """

    component_spec_path: str
    arguments: t.Dict[str, t.Any]
    number_of_gpus: t.Optional[int] = None
    node_pool_name: t.Optional[str] = None
    p_volumes: t.Optional[t.Dict[str, k8s_client.V1Volume]] = None
    ephemeral_storage_size: t.Optional[str] = None

    def __post_init__(self):
        with open(self.component_spec_path, encoding="utf-8") as file_:
            specification = yaml.safe_load(file_)
        super().__init__(specification)


class FondantPipeline:
    """Class representing a Fondant Pipeline"""

    def __init__(self, host: str):
        """
        Args:
            host: The `host` URL argument specifies the Kubeflow Pipelines API endpoint to
             which the client should send requests.
        """
        self.client = kfp.Client(host=host)

    def get_pipeline_id(self, pipeline_name: str) -> str:
        """
        Function that returns the id of a pipeline given a pipeline name
        Args:
            pipeline_name: the name of the pipeline
        Returns:
            The pipeline id
        """
        return self.client.get_pipeline_id(pipeline_name)

    def get_pipeline_version_ids(self, pipeline_id: str) -> t.List[str]:
        """Function that returns the versions of a pipeline given a pipeline id"""
        pipeline_versions = self.client.list_pipeline_versions(pipeline_id).versions
        return [getattr(version, "id") for version in pipeline_versions]

    @staticmethod
    def _validate_pipeline_definition(
        fondant_components_operation: t.List[FondantComponentOperation],
    ):
        """
        Validates the pipeline definition by ensuring that the input and output subsets of each
         component match and are invoked in the correct order

        Raises:
            InvalidPipelineDefinition: If a component is trying to invoke a subset that is not
             defined or created in previous components, or if an invoked subset's schema does not
              match the previously created subset definition.
        """
        load_component = True
        available_subsets: t.Dict[str, ComponentSubset] = {}
        for fondant_component_operation in fondant_components_operation:
            if not load_component:
                for (
                    subset_name,
                    subset,
                ) in fondant_component_operation.input_subsets.items():
                    if subset_name not in available_subsets:
                        raise InvalidPipelineDefinition(
                            f"Component '{fondant_component_operation.name}' "
                            f"is trying to invoke the subset '{subset_name}', "
                            f"which has not been defined or created in the previous components."
                        )
                    if subset.fields != available_subsets[subset_name].fields:
                        raise InvalidPipelineDefinition(
                            f"The invoked subset '{subset_name}' of the"
                            f" '{fondant_component_operation.name}' component does not match the"
                            f" previously created subset definition.\n The '{subset_name}' schema "
                            f"is currently defined with the following fields:\n"
                            f"{available_subsets[subset_name].fields}\n"
                            f"The current component to trying to invoke it with this schema:\n"
                            f"{subset.fields}"
                        )
            else:
                available_subsets.update(fondant_component_operation.input_subsets)
            available_subsets.update(fondant_component_operation.output_subsets)
            load_component = False
        logger.info("All pipeline component specifications match.")

    def compile_pipeline(
        self,
        *,
        pipeline_name: str,
        pipeline_description: str,
        fondant_components_operation: t.List[FondantComponentOperation],
        pipeline_package_path: str,
    ):
        """
        Function that creates and compiles a Kubeflow Pipeline.

        Args:
            pipeline_name: The name of the pipeline.
            pipeline_description: A brief description of the pipeline.
            fondant_components_operation: A list of FondantComponent operations that define
             components used in the pipeline. The operations must be ordered in the order of
             execution.
            pipeline_package_path (str): The path to the directory where the pipeline package will
             be generated.

        Example usage:
            To create a pipeline, first define the pipeline components as fondantComponentOperation
             objects, then call this function to compile the pipeline function:

            fondant_components_operations = [
                FondantComponentOperation(yaml_spec_path=<path_to_yaml>, args:{batch_size: 8, ...}),
                FondantComponentOperation(...),
            ]

            pipeline_function = create_pipeline(
                name='MyPipeline',
                 description='A description of my pipeline',
                fondant_components_operations=fondant_components_operations,
                pipeline_package_path='/path/to/pipeline/package/package.tgz'
            )

            Once you have generated the pipeline function, you can use it to create an instance of
            the pipeline and compile it using the Kubeflow compiler.
        """

        def _get_component_function(
            fondant_component_operation: FondantComponentOperation,
        ) -> t.Callable:
            """
            Load the Kubeflow component based on the specification from the fondant component
             operation.
            Args:
                fondant_component_operation (FondantComponentOperation): The fondant component
                 operation.
            Returns:
                Callable: The Kubeflow component.
            """
            return kfp.components.load_component(
                text=fondant_component_operation.kubeflow_specification.to_text()
            )

        def _set_task_configuration(task, fondant_component_operation):
            # Unpack optional specifications
            number_of_gpus = fondant_component_operation.number_of_gpus
            node_pool_name = fondant_component_operation.node_pool_name
            p_volumes = fondant_component_operation.p_volumes
            ephemeral_storage_size = fondant_component_operation.ephemeral_storage_size

            # Assign optional specification
            if number_of_gpus is not None:
                task.set_gpu_limit(number_of_gpus)
            if node_pool_name is not None:
                task.add_node_selector_constraint("node_pool", node_pool_name)
            if p_volumes is not None:
                task.add_pvolumes(p_volumes)
            if ephemeral_storage_size is not None:
                task.set_ephemeral_storage_request(ephemeral_storage_size)

            return task

        # Validate subset schema before defining the pipeline
        self._validate_pipeline_definition(fondant_components_operation)

        @dsl.pipeline(name=pipeline_name, description=pipeline_description)
        def pipeline():
            # TODO: check if we want to have the manifest path empty for loading component or remove
            #  it completely from the loading component
            manifest_path = ""
            previous_component_task = None
            for fondant_component_operation in fondant_components_operation:
                # Get the Kubeflow component based on the fondant component operation.
                component_op = _get_component_function(fondant_component_operation)

                # Execute the Kubeflow component and pass in the output manifest path from
                # the previous component.
                component_args = fondant_component_operation.arguments
                component_task = component_op(
                    input_manifest_path=manifest_path, **component_args
                )

                # Set optional configurations
                component_task = _set_task_configuration(
                    component_task, fondant_component_operation
                )
                # Set the execution order of the component task to be after the previous
                # component task.
                if previous_component_task is not None:
                    component_task.after(previous_component_task)

                # Update the manifest path to be the output path of the current component task.
                manifest_path = component_task.outputs["output_manifest_path"]

                previous_component_task = component_task

            return pipeline

        kfp.compiler.Compiler().compile(pipeline, pipeline_package_path)

        logger.info("Pipeline compiled successfully")

    def upload_pipeline(
        self,
        *,
        pipeline_name: str,
        pipeline_package_path: str,
        delete_pipeline_package: t.Optional[bool] = False,
    ):
        """
        Uploads a pipeline package to Kubeflow Pipelines and deletes any existing pipeline with the
         same name.
        Args:
            pipeline_name: The name of the pipeline.
            pipeline_package_path: The path to the pipeline package tarball (.tar.gz) file.
            delete_pipeline_package: Whether to delete the pipeline package file
             after uploading. Defaults to False.
        Raises:
            Exception: If there was an error uploading the pipeline package.
        """

        self.delete_pipeline(pipeline_name)
        logger.info(f"Uploading pipeline: {pipeline_name}")

        try:
            self.client.upload_pipeline(
                pipeline_package_path=pipeline_package_path, pipeline_name=pipeline_name
            )
        except Exception as e:
            raise Exception(f"Error uploading pipeline package: {str(e)}")

        # Delete the pipeline package file if specified.
        if delete_pipeline_package:
            Path(pipeline_package_path).unlink()

    def delete_pipeline(self, pipeline_name: str):
        """
        Function that deletes the pipeline name
        Args:
            pipeline_name: the name of the pipeline to delete
        """
        pipeline_id = self.get_pipeline_id(pipeline_name)
        if pipeline_id is not None:
            pipeline_version_ids = self.get_pipeline_version_ids(pipeline_id)
            # All versions need to be first deleted
            for pipeline_version_id in pipeline_version_ids:
                self.client.delete_pipeline_version(pipeline_version_id)
            self.client.delete_pipeline(pipeline_id)

            logger.info(
                f"Pipeline {pipeline_name} already exists. Deleting old pipeline..."
            )
        else:
            logger.info(f"No existing pipeline under `{pipeline_name}` name was found.")

    def run_pipeline(self):
        raise NotImplementedError
