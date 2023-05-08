"""This module defines classes to represent a Fondant Pipeline."""
import json
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import yaml

from fondant.component import FondantComponentSpec, Manifest
from fondant.exceptions import InvalidPipelineDefinition
from fondant.import_utils import is_kfp_available

if is_kfp_available():
    import kfp
    from kfp import dsl
    from kubernetes import client as k8s_client

logger = logging.getLogger(__name__)


@dataclass
class FondantComponentOp(FondantComponentSpec):
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

    def __name__(self):
        return self.name


class FondantClient:
    """Class representing a Fondant Client."""

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
            The pipeline id.
        """
        return self.client.get_pipeline_id(pipeline_name)

    def get_pipeline_version_ids(self, pipeline_id: str) -> t.List[str]:
        """Function that returns the versions of a pipeline given a pipeline id."""
        pipeline_versions = self.client.list_pipeline_versions(pipeline_id).versions
        return [getattr(version, "id") for version in pipeline_versions]

    def delete_pipeline(self, pipeline_name: str):
        """
        Function that deletes the pipeline name
        Args:
            pipeline_name: the name of the pipeline to delete.
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

    def upload_pipeline(
        self,
        *,
        pipeline_name: str,
        pipeline_package_path: t.Optional[str] = None,
        delete_pipeline_package: t.Optional[bool] = False,
    ):
        """
        Uploads a pipeline package to Kubeflow Pipelines and deletes any existing pipeline with the
         same name.

        Args:
            pipeline_name: The name of the pipeline.
            pipeline_package_path: The path to the compiled package path.
            delete_pipeline_package: Whether to delete the pipeline package file
             after uploading. Defaults to False.

        Raises:
            Exception: If there was an error uploading the pipeline package.
        """
        # self.delete_pipeline(pipeline_name)
        logger.info(f"Uploading pipeline: {pipeline_name}")

        # Get pipeline package path
        if pipeline_package_path is None:
            pipeline_package_path = f"{pipeline_name}.tgz"
        try:
            self.client.upload_pipeline(
                pipeline_package_path=pipeline_package_path, pipeline_name=pipeline_name
            )
        except Exception as e:
            raise Exception(f"Error uploading pipeline package: {str(e)}")

        # Delete the pipeline package file if specified.
        if delete_pipeline_package:
            Path(pipeline_package_path).unlink()

    def run_pipeline(self):
        raise NotImplementedError


class FondantPipeline:
    """Class representing a Fondant Pipeline."""

    def __init__(
        self,
        base_path: str,
        pipeline_name: str,
        pipeline_description: t.Optional[str] = None,
    ):
        """
        Initialize the FondantPipeline object.

        Args:
            base_path: The base path for the pipeline where the artifacts are stored.
            pipeline_name: The name of the pipeline.
            pipeline_description: Optional description of the pipeline.
        """
        self.base_path = base_path
        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description
        self._graph: t.Dict[str, t.Any] = {}

    def add_op(
        self,
        task: FondantComponentOp,
        dependency: t.Optional[FondantComponentOp] = None,
    ):
        """
        Add a task to the pipeline with an optional dependency.

        Args:
            task: The task to add to the pipeline.
            dependency: Optional dependency task that needs to be completed before the task can run.
        """
        # TODO allow dependencies to be a list of operations instead of one
        if dependency is None:
            dependencies = []
        else:
            dependencies = [dependency.name]
        task_name = task.name
        self._graph[task_name] = {
            "fondant_component_op": task,
            "dependencies": dependencies,
        }

    def sort_graph(self):
        """Sort the graph topologically based on task dependencies."""
        sorted_graph = []
        visited = set()

        def depth_first_traversal(node: str):
            """
            Perform a depth-first traversal of the graph and its dependencies.

            Args:
                node: The name of the starting node for traversal.
            """
            if node not in visited:
                visited.add(node)
                for dependency in self._graph[node]["dependencies"]:
                    depth_first_traversal(dependency)
                sorted_graph.append(node)

        for node in self._graph:
            depth_first_traversal(node)

        self._graph = {node: self._graph[node] for node in sorted_graph}

    def __repr__(self) -> str:
        """Return a string representation of the FondantPipeline object."""
        return f"{self.__class__.__name__}({self._graph!r}"

    def _validate_pipeline_definition(self, run_id: str):
        """
        Validates the pipeline definition by ensuring that the input and output subsets and their
         associated fields match and are invoked in the correct order.

        Raises:
            InvalidPipelineDefinition: If a component is trying to invoke a subset that is not
             defined or created in previous components, or if an invoked subset's schema does not
              match the previously created subset definition.
            base_path: the base path where to store the pipelines artifacts
            run_id: the run id of the component
        """
        load_component = True
        load_component_name = list(self._graph.keys())[0]

        # Create initial manifest
        manifest = Manifest.create(
            base_path=self.base_path, run_id=run_id, component_id=load_component_name
        )
        for _, operation_specs in self._graph.items():
            fondant_component_op = operation_specs["fondant_component_op"]
            component_spec = FondantComponentSpec.from_file(
                fondant_component_op.component_spec_path
            )
            manifest = manifest.evolve(component_spec)
            if not load_component:
                # Check subset exists
                for (
                    component_subset_name,
                    component_subset,
                ) in fondant_component_op.input_subsets.items():
                    manifest_subset = manifest.subsets
                    if component_subset_name not in manifest_subset:
                        raise InvalidPipelineDefinition(
                            f"Component '{fondant_component_op.name}' "
                            f"is trying to invoke the subset '{component_subset_name}', "
                            f"which has not been defined or created in the previous components."
                        )

                    # Get the corresponding manifest fields
                    manifest_fields = manifest_subset[component_subset_name].fields

                    # Check fields
                    for field_name, subset_field in component_subset.fields.items():
                        # Check if invoked field exists
                        if field_name not in manifest_fields:
                            raise InvalidPipelineDefinition(
                                f"The invoked subset '{component_subset_name}' of the"
                                f" '{fondant_component_op.name}' component does not match "
                                f"the previously created subset definition.\n The component is"
                                f" trying to invoke the field '{field_name}' which has not been"
                                f" previously defined. Current available fields are "
                                f"{manifest_fields}\n"
                            )
                        # Check if the invoked field schema matches the current schema
                        if subset_field != manifest_fields[field_name]:
                            raise InvalidPipelineDefinition(
                                f"The invoked subset '{component_subset_name}' of the"
                                f" '{fondant_component_op.name}' component does not match "
                                f" the previously created subset definition.\n The '{field_name}'"
                                f" field is currently defined with the following schema:\n"
                                f"{manifest_fields[field_name]}\n"
                                f"The current component to trying to invoke it with this schema:\n"
                                f"{subset_field}"
                            )
            load_component = False

        logger.info("All pipeline component specifications match.")

    def compile_pipeline(self, pipeline_package_path: t.Optional[str] = None):
        """
        Function that creates and compiles a Kubeflow Pipeline.
        Once you have generated the pipeline function, you can use it to create an instance of
        the pipeline and compile it using the Kubeflow compiler.
        """

        def _get_component_function(
            fondant_component_operation: FondantComponentOp,
        ) -> t.Callable:
            """
            Load the Kubeflow component based on the specification from the fondant component
             operation.

            Args:
                fondant_component_operation (FondantComponentOp): The fondant component
                 operation.

            Returns:
                Callable: The Kubeflow component.
            """
            return kfp.components.load_component(
                text=fondant_component_operation.kubeflow_specification.to_string()
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

        # Sort graph based on specified dependencies
        self.sort_graph()

        # Get pipeline package path
        if pipeline_package_path is None:
            pipeline_package_path = f"{self.pipeline_name}.tgz"

        # parse metadata argument required for the first component
        run_id = "{{workflow.name}}"

        # Validate subset schema before defining the pipeline
        self._validate_pipeline_definition(run_id)

        @dsl.pipeline(name=self.pipeline_name, description=self.pipeline_description)
        def pipeline():
            # TODO: check if we want to have the manifest path empty for loading component or remove
            #  it completely from the loading component
            # TODO: check if we want to have the metadata arg empty for transform component or
            #  remove it completely from the transform component
            manifest_path = ""
            metadata = ""
            previous_component_task = None
            for _, operation_specs in self._graph.items():
                fondant_component_op = operation_specs["fondant_component_op"]
                # Get the Kubeflow component based on the fondant component operation.
                kubeflow_component_op = _get_component_function(fondant_component_op)

                # Execute the Kubeflow component and pass in the output manifest path from
                # the previous component.
                component_args = fondant_component_op.arguments
                if previous_component_task is not None:
                    component_task = kubeflow_component_op(
                        input_manifest_path=manifest_path,
                        metadata=metadata,
                        **component_args,
                    )
                else:
                    metadata = json.dumps(
                        {"base_path": self.base_path, "run_id": run_id}
                    )
                    # Add metadata to the first component
                    component_task = kubeflow_component_op(
                        input_manifest_path=manifest_path,
                        metadata=metadata,
                        **component_args,
                    )
                    metadata = ""
                # Set optional configurations
                component_task = _set_task_configuration(
                    component_task, fondant_component_op
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
