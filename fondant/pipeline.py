"""This module defines classes to represent a Fondant Pipeline."""
import json
import logging
import typing as t
from collections import OrderedDict
from pathlib import Path

try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from fondant.component import ComponentSpec, Manifest
from fondant.exceptions import InvalidPipelineDefinition
from fondant.import_utils import is_kfp_available

if is_kfp_available():
    import kfp
    from kfp import dsl
    from kubernetes import client as k8s_client

logger = logging.getLogger(__name__)


class ComponentOp:
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
        ephemeral_storage_size: Used ephemeral-storage size (minimum) for the operation.
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

    def __init__(
        self,
        component_spec_path: t.Union[str, Path],
        *,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        number_of_gpus: t.Optional[int] = None,
        node_pool_name: t.Optional[str] = None,
        p_volumes: t.Optional[t.Dict[str, k8s_client.V1Volume]] = None,
        ephemeral_storage_size: t.Optional[str] = None,
    ):
        self.component_spec_path = component_spec_path
        self.arguments = arguments or {}
        self.number_of_gpus = number_of_gpus
        self.node_pool_name = node_pool_name
        self.p_volumes = p_volumes
        self.ephemeral_storage_size = ephemeral_storage_size

    @property
    def component_spec(self) -> ComponentSpec:
        return ComponentSpec.from_file(self.component_spec_path)

    @classmethod
    def from_registry(
        cls,
        name: str,
        *,
        arguments: t.Optional[t.Dict[str, t.Any]] = None,
        number_of_gpus: t.Optional[int] = None,
        node_pool_name: t.Optional[str] = None,
        p_volumes: t.Optional[t.Dict[str, k8s_client.V1Volume]] = None,
        ephemeral_storage_size: t.Optional[str] = None,
    ) -> "ComponentOp":
        """Load a reusable component by its name.

        Args:
            name: Name of the component to load
            arguments: A dictionary containing the argument name and value for the operation.
            number_of_gpus: The number of gpus to assign to the operation
            node_pool_name: The name of the node pool to which the operation will be assigned.
            p_volumes: Collection of persistent volumes in a Kubernetes cluster. Keys are mount
                paths, values are Kubernetes volumes or inherited types(e.g. PipelineVolumes).
            ephemeral_storage_size: Used ephemeral-storage request (minimum) for the operation.
                Defined by string which can be a number or a number followed by one of “E”, “P”,
                “T”, “G”, “M”, “K”. (e.g. 2T for 2 Terabytes)
        """
        component_spec_path = (
            files("fondant") / f"components/{name}/fondant_component.yaml"
        )
        component_spec_path = t.cast(Path, component_spec_path)

        if not (component_spec_path.exists() and component_spec_path.is_file()):
            raise ValueError(f"No reusable component with name {name} found.")

        return ComponentOp(
            component_spec_path,
            arguments=arguments,
            number_of_gpus=number_of_gpus,
            node_pool_name=node_pool_name,
            p_volumes=p_volumes,
            ephemeral_storage_size=ephemeral_storage_size,
        )


class Pipeline:
    """Class representing a Fondant Pipeline."""

    def __init__(
        self,
        base_path: str,
        pipeline_name: str,
        pipeline_description: t.Optional[str] = None,
    ):
        """
        Args:
            base_path: The base path for the pipeline where the artifacts are stored.
            pipeline_name: The name of the pipeline.
            pipeline_description: Optional description of the pipeline.
        """
        self.base_path = base_path
        self.name = pipeline_name
        self.description = pipeline_description
        self.package_path = f"{pipeline_name}.tgz"
        self._graph: t.OrderedDict[str, t.Any] = OrderedDict()
        self.task_without_dependencies_added = False

    def add_op(
        self,
        task: ComponentOp,
        dependencies: t.Optional[t.Union[ComponentOp, t.List[ComponentOp]]] = None,
    ):
        """
        Add a task to the pipeline with an optional dependency.

        Args:
            task: The task to add to the pipeline.
            dependencies: Optional task dependencies that needs to be completed before the task
             can run.
        """
        if dependencies is None:
            if self.task_without_dependencies_added:
                raise InvalidPipelineDefinition(
                    "At most one task can be defined without " "dependencies."
                )
            dependencies = []
            self.task_without_dependencies_added = True
        elif not isinstance(dependencies, list):
            dependencies = [dependencies]

        if len(dependencies) > 1:
            raise InvalidPipelineDefinition(
                f"Multiple component dependencies provided for component"
                f" `{task.component_spec.name}`. "
                f"The current version of Fondant can only handle components with a single "
                f"dependency. Please note that the behavior of the pipeline may be unpredictable"
                f" or incorrect."
            )

        dependencies_names = [
            dependency.component_spec.name for dependency in dependencies
        ]

        self._graph[task.component_spec.name] = {
            "fondant_component_op": task,
            "dependencies": dependencies_names,
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

        for graph_node in self._graph:
            depth_first_traversal(graph_node)

        self._graph = OrderedDict((node, self._graph[node]) for node in sorted_graph)

    def _validate_pipeline_definition(self, run_id: str):
        """
        Validates the pipeline definition by ensuring that the consumed and produced subsets and
        their associated fields match and are invoked in the correct order.

        Raises:
            InvalidPipelineDefinition: If a component is trying to invoke a subset that is not
             defined or created in previous components, or if an invoked subset's schema does not
              match the previously created subset definition.
            base_path: the base path where to store the pipelines artifacts
            run_id: the run id of the component
        """
        # TODO: change later if we decide to run 2 fondan pipelines after each other
        load_component = True
        load_component_name = list(self._graph.keys())[0]

        # Create initial manifest
        manifest = Manifest.create(
            base_path=self.base_path, run_id=run_id, component_id=load_component_name
        )
        for operation_specs in self._graph.values():
            fondant_component_op = operation_specs["fondant_component_op"]
            component_spec = fondant_component_op.component_spec
            if not load_component:
                # Check subset exists
                for (
                    component_subset_name,
                    component_subset,
                ) in component_spec.consumes.items():
                    if component_subset_name not in manifest.subsets:
                        raise InvalidPipelineDefinition(
                            f"Component '{component_spec.name}' "
                            f"is trying to invoke the subset '{component_subset_name}', "
                            f"which has not been defined or created in the previous components."
                        )

                    # Get the corresponding manifest fields
                    manifest_fields = manifest.subsets[component_subset_name].fields

                    # Check fields
                    for field_name, subset_field in component_subset.fields.items():
                        # Check if invoked field exists
                        if field_name not in manifest_fields:
                            raise InvalidPipelineDefinition(
                                f"The invoked subset '{component_subset_name}' of the"
                                f" '{component_spec.name}' component does not match "
                                f"the previously created subset definition.\n The component is"
                                f" trying to invoke the field '{field_name}' which has not been"
                                f" previously defined. Current available fields are "
                                f"{manifest_fields}\n"
                            )
                        # Check if the invoked field schema matches the current schema
                        if subset_field != manifest_fields[field_name]:
                            raise InvalidPipelineDefinition(
                                f"The invoked subset '{component_subset_name}' of the"
                                f" '{component_spec.name}' component does not match "
                                f" the previously created subset definition.\n The '{field_name}'"
                                f" field is currently defined with the following schema:\n"
                                f"{manifest_fields[field_name]}\n"
                                f"The current component to trying to invoke it with this schema:\n"
                                f"{subset_field}"
                            )
            manifest = manifest.evolve(component_spec)
            load_component = False

        logger.info("All pipeline component specifications match.")

    def compile(self):
        """
        Function that creates and compiles a Kubeflow Pipeline.
        Once you have generated the pipeline function, you can use it to create an instance of
        the pipeline and compile it using the Kubeflow compiler.
        """

        def _get_component_function(
            fondant_component_operation: ComponentOp,
        ) -> t.Callable:
            """
            Load the Kubeflow component based on the specification from the fondant component
             operation.

            Args:
                fondant_component_operation (ComponentOp): The fondant component
                 operation.

            Returns:
                Callable: The Kubeflow component.
            """
            return kfp.components.load_component(
                text=fondant_component_operation.component_spec.kubeflow_specification.to_string()
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

        # parse metadata argument required for the first component
        run_id = "{{workflow.name}}"

        # Validate subset schema before defining the pipeline
        self._validate_pipeline_definition(run_id)

        @dsl.pipeline(name=self.name, description=self.description)
        def pipeline():
            # TODO: check if we want to have the manifest path empty for loading component or remove
            #  it completely from the loading component
            # TODO: check if we want to have the metadata arg empty for transform component or
            #  remove it completely from the transform component
            manifest_path = ""
            metadata = ""
            previous_component_task = None
            for operation in self._graph.values():
                fondant_component_op = operation["fondant_component_op"]
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

        logger.info(f"Compiling pipeline: {self.name}")

        kfp.compiler.Compiler().compile(pipeline, self.package_path)

        logger.info("Pipeline compiled successfully")

    def __repr__(self) -> str:
        """Return a string representation of the FondantPipeline object."""
        return f"{self.__class__.__name__}({self._graph!r}"


class Client:
    """Class representing a Fondant Client."""

    def __init__(self, host: str):
        """
        Args:
            host: The `host` URL argument specifies the Kubeflow Pipelines API endpoint to
             which the client should send requests.
        """
        self.host = host
        self.client = kfp.Client(host=self.host)

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

    def compile_and_upload(
        self,
        pipeline: Pipeline,
        delete_pipeline_package: t.Optional[bool] = False,
    ):
        """
        Uploads a pipeline package to Kubeflow Pipelines and deletes any existing pipeline with the
         same name.

        Args:
            pipeline: The fondant pipeline to compile and upload
            delete_pipeline_package: Whether to delete the pipeline package file
             after uploading. Defaults to False.

        Raises:
            Exception: If there was an error uploading the pipeline package.
        """
        # self.delete_pipeline(pipeline.name)

        pipeline.compile()

        logger.info(f"Uploading pipeline: {pipeline.name}")

        try:
            self.client.upload_pipeline(
                pipeline_package_path=pipeline.package_path, pipeline_name=pipeline.name
            )
        except Exception as e:
            raise Exception(f"Error uploading pipeline package: {str(e)}")

        # Delete the pipeline package file if specified.
        if delete_pipeline_package:
            Path(pipeline.package_path).unlink()

    def compile_and_run(
        self,
        pipeline: Pipeline,
        run_name: t.Optional[str] = None,
        experiment_name: t.Optional[str] = "Default",
    ):
        """
        Compiles and runs the specified pipeline.

        Args:
            pipeline: The pipeline object to be compiled and run.
            run_name: The name of the run. If not provided, the pipeline's name with 'run' appended
             will be used.
            experiment_name: The name of the experiment where the pipeline will be run.
            Default is 'Default'.
        """
        pipeline.compile()

        try:
            experiment = self.client.get_experiment(experiment_name=experiment_name)
        except ValueError:
            logger.info(
                f"Defined experiment '{experiment_name}' not found. Creating new experiment"
                f"under this name"
            )
            experiment = self.client.create_experiment(experiment_name)

        if run_name is None:
            run_name = pipeline.name + " run"

        pipeline_spec = self.client.run_pipeline(
            experiment_id=experiment.id,
            job_name=run_name,
            pipeline_package_path=pipeline.package_path,
        )

        pipeline_url = f"{self.host}/#/runs/details/{pipeline_spec.id}"
        logger.info(f"Pipeline is running at: {pipeline_url}")
