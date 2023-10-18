import json
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from fondant.exceptions import InvalidPipelineDefinition
from fondant.manifest import Metadata
from fondant.pipeline import (
    Pipeline,
    valid_accelerator_types,
    valid_vertex_accelerator_types,
)
from fondant.schema import KubeflowCommandArguments  # noqa: TCH001

logger = logging.getLogger(__name__)

DASK_DIAGNOSTIC_DASHBOARD_PORT = 8787


class Compiler(ABC):
    """Abstract base class for a compiler."""

    @abstractmethod
    def compile(self, *args, **kwargs) -> None:
        """Abstract method to invoke compilation."""

    @abstractmethod
    def _set_configuration(self, *args, **kwargs) -> None:
        """Abstract method to set pipeline configuration."""


@dataclass
class DockerVolume:
    """Dataclass representing a DockerVolume.
    (https://docs.docker.com/compose/compose-file/05-services/#volumes).

    Args:
        type: the mount type volume (bind, volume)
        source: the source of the mount, a path on the host for a bind mount
        target: the path in the container where the volume is mounted.
    """

    type: str
    source: str
    target: str


class DockerCompiler(Compiler):
    """Compiler that creates a docker-compose spec from a pipeline."""

    def compile(
        self,
        pipeline: Pipeline,
        *,
        output_path: str = "docker-compose.yml",
        extra_volumes: t.Optional[list] = None,
        build_args: t.Optional[t.List[str]] = None,
    ) -> None:
        """Compile a pipeline to docker-compose spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the docker-compose spec
            extra_volumes: a list of extra volumes (using the Short syntax:
              https://docs.docker.com/compose/compose-file/05-services/#short-syntax-5)
              to mount in the docker-compose spec.
            build_args: List of build arguments to pass to docker
        """
        if extra_volumes is None:
            extra_volumes = []

        logger.info(f"Compiling {pipeline.name} to {output_path}")
        spec = self._generate_spec(
            pipeline,
            extra_volumes=extra_volumes,
            build_args=build_args or [],
        )

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        with open(output_path, "w") as outfile:
            yaml.dump(spec, outfile, Dumper=NoAliasDumper, default_flow_style=False)

        logger.info(f"Successfully compiled to {output_path}")

    @staticmethod
    def _patch_path(base_path: str) -> t.Tuple[str, t.Optional[DockerVolume]]:
        """Helper that checks if the base_path is local or remote,
        if local it patches the base_path and prepares a bind mount
        Returns a tuple containing the path and volume.
        """
        p_base_path = Path(base_path)
        # check if base path is an existing local folder
        if p_base_path.exists():
            logger.info(
                f"Base path found on local system, setting up {base_path} as mount volume",
            )
            p_base_path = p_base_path.resolve()
            volume = DockerVolume(
                type="bind",
                source=str(p_base_path),
                target=f"/{p_base_path.stem}",
            )
            path = f"/{p_base_path.stem}"
        else:
            logger.info(f"Base path {base_path} is remote")
            volume = None
            path = base_path
        return path, volume

    def _generate_spec(
        self,
        pipeline: Pipeline,
        *,
        extra_volumes: t.List[str],
        build_args: t.List[str],
    ) -> dict:
        """Generate a docker-compose spec as a python dictionary,
        loops over the pipeline graph to create services and their dependencies.
        """
        path, volume = self._patch_path(base_path=pipeline.base_path)
        run_id = pipeline.get_run_id()

        services = {}

        pipeline.validate(run_id=run_id)

        component_cache_key = None

        for component_name, component in pipeline._graph.items():
            component_op = component["fondant_component_op"]

            component_cache_key = component_op.get_component_cache_key(
                previous_component_cache=component_cache_key,
            )

            metadata = Metadata(
                pipeline_name=pipeline.name,
                run_id=run_id,
                base_path=path,
                component_id=component_name,
                cache_key=component_cache_key,
            )

            logger.info(f"Compiling service for {component_name}")

            # add metadata argument to command
            command = ["--metadata", metadata.to_json()]

            # add in and out manifest paths to command
            command.extend(
                [
                    "--output_manifest_path",
                    f"{path}/{metadata.pipeline_name}/{metadata.run_id}/"
                    f"{component_name}/manifest.json",
                ],
            )

            # add arguments if any to command
            for key, value in component_op.arguments.items():
                if isinstance(value, (dict, list)):
                    command.extend([f"--{key}", json.dumps(value)])
                else:
                    command.extend([f"--{key}", f"{value}"])

            # resolve dependencies
            depends_on = {}
            if component["dependencies"]:
                for dependency in component["dependencies"]:
                    depends_on[dependency] = {
                        "condition": "service_completed_successfully",
                    }
                    # there is only an input manifest if the component has dependencies
                    command.extend(
                        [
                            "--input_manifest_path",
                            f"{path}/{metadata.pipeline_name}/{metadata.run_id}/"
                            f"{dependency}/manifest.json",
                        ],
                    )

            volumes: t.List[t.Union[str, dict]] = []
            if volume:
                volumes.append(asdict(volume))
            if extra_volumes:
                volumes.extend(extra_volumes)

            ports: t.List[t.Union[str, dict]] = []
            ports.append(
                f"{DASK_DIAGNOSTIC_DASHBOARD_PORT}:{DASK_DIAGNOSTIC_DASHBOARD_PORT}",
            )

            services[component_name] = {
                "command": command,
                "depends_on": depends_on,
                "volumes": volumes,
                "ports": ports,
            }

            self._set_configuration(services, component_op, component_name)

            if component_op.dockerfile_path is not None:
                logger.info(
                    f"Found Dockerfile for {component_name}, adding build step.",
                )
                services[component_name]["build"] = {
                    "context": str(component_op.component_dir),
                    "args": build_args,
                }
            else:
                services[component_name]["image"] = component_op.component_spec.image

        return {
            "name": pipeline.name,
            "version": "3.8",
            "services": services,
        }

    @staticmethod
    def _set_configuration(services, fondant_component_operation, component_name):
        accelerator_name = fondant_component_operation.accelerator_name
        accelerator_number = fondant_component_operation.number_of_accelerators

        if accelerator_name is not None:
            if accelerator_name not in valid_accelerator_types:
                msg = (
                    f"Configured accelerator `{accelerator_name}`"
                    f" is not a valid accelerator type for Docker Compose compiler."
                    f" Available options: {valid_vertex_accelerator_types}"
                )
                raise InvalidPipelineDefinition(msg)

            if accelerator_name == "GPU":
                services[component_name]["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": accelerator_number,
                                    "capabilities": ["gpu"],
                                },
                            ],
                        },
                    },
                }
            elif accelerator_name == "TPU":
                msg = "TPU configuration is not yet implemented for Docker Compose "
                raise NotImplementedError(msg)

        return services


class KubeFlowCompiler(Compiler):
    """Compiler that creates a Kubeflow pipeline spec from a pipeline."""

    def __init__(self):
        self._resolve_imports()

    def _resolve_imports(self):
        """Resolve imports for the Kubeflow compiler."""
        try:
            import kfp
            import kfp.kubernetes as kfp_kubernetes

            self.kfp = kfp
            self.kfp_kubernetes = kfp_kubernetes

        except ImportError:
            msg = """You need to install kfp to use the Kubeflow compiler,\n
                     you can install it with `pip install fondant[kfp]`"""
            raise ImportError(
                msg,
            )

    def compile(
        self,
        pipeline: Pipeline,
        output_path: str,
    ) -> None:
        """Compile a pipeline to Kubeflow pipeline spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the Kubeflow pipeline spec
        """
        run_id = pipeline.get_run_id()
        pipeline.validate(run_id=run_id)
        logger.info(f"Compiling {pipeline.name} to {output_path}")

        def set_component_exec_args(
            *,
            component_op,
            component_args: t.List[str],
            input_manifest_path: bool,
        ):
            """Dump Fondant specification arguments to kfp command executor arguments."""
            dumped_args: KubeflowCommandArguments = []

            component_args.extend(["output_manifest_path", "metadata"])
            if input_manifest_path:
                component_args.append("input_manifest_path")

            for arg in component_args:
                arg_name = arg.strip().replace(" ", "_")
                arg_name_cmd = f"--{arg_name}"

                dumped_args.append(arg_name_cmd)
                dumped_args.append("{{$.inputs.parameters['" + f"{arg_name}" + "']}}")

            component_op.component_spec.implementation.container.args = dumped_args

            return component_op

        @self.kfp.dsl.pipeline(name=pipeline.name, description=pipeline.description)
        def kfp_pipeline():
            previous_component_task = None
            component_cache_key = None

            for component_name, component in pipeline._graph.items():
                logger.info(f"Compiling service for {component_name}")

                component_op = component["fondant_component_op"]
                # convert ComponentOp to Kubeflow component
                kubeflow_component_op = self.kfp.components.load_component_from_text(
                    text=component_op.component_spec.kubeflow_specification.to_string(),
                )

                # Remove None values from arguments
                component_args = {
                    k: v for k, v in component_op.arguments.items() if v is not None
                }

                component_cache_key = component_op.get_component_cache_key(
                    previous_component_cache=component_cache_key,
                )
                metadata = Metadata(
                    pipeline_name=pipeline.name,
                    run_id=run_id,
                    base_path=pipeline.base_path,
                    component_id=component_name,
                    cache_key=component_cache_key,
                )

                output_manifest_path = (
                    f"{pipeline.base_path}/{pipeline.name}/"
                    f"{run_id}/{component_name}/manifest.json"
                )
                # Set the execution order of the component task to be after the previous
                # component task.
                if component["dependencies"]:
                    for dependency in component["dependencies"]:
                        input_manifest_path = (
                            f"{pipeline.base_path}/{pipeline.name}/"
                            f"{run_id}/{dependency}/manifest.json"
                        )
                        kubeflow_component_op = set_component_exec_args(
                            component_op=kubeflow_component_op,
                            component_args=list(component_args.keys()),
                            input_manifest_path=True,
                        )
                        component_task = kubeflow_component_op(
                            input_manifest_path=input_manifest_path,
                            output_manifest_path=output_manifest_path,
                            metadata=metadata.to_json(),
                            **component_args,
                        )
                        component_task.after(previous_component_task)

                else:
                    kubeflow_component_op = set_component_exec_args(
                        component_op=kubeflow_component_op,
                        component_args=list(component_args.keys()),
                        input_manifest_path=False,
                    )
                    component_task = kubeflow_component_op(
                        metadata=metadata.to_json(),
                        output_manifest_path=output_manifest_path,
                        **component_args,
                    )

                # Set optional configuration
                component_task = self._set_configuration(
                    component_task,
                    component_op,
                )

                # Disable caching
                component_task.set_caching_options(enable_caching=False)

                previous_component_task = component_task

        logger.info(f"Compiling {pipeline.name} to {output_path}")

        self.kfp.compiler.Compiler().compile(kfp_pipeline, output_path)  # type: ignore
        logger.info("Pipeline compiled successfully")

    def _set_configuration(self, task, fondant_component_operation):
        # Unpack optional specifications
        number_of_accelerators = fondant_component_operation.number_of_accelerators
        accelerator_name = fondant_component_operation.accelerator_name
        node_pool_label = fondant_component_operation.node_pool_label
        node_pool_name = fondant_component_operation.node_pool_name
        cpu_request = fondant_component_operation.cpu_request
        cpu_limit = fondant_component_operation.cpu_limit
        memory_request = fondant_component_operation.memory_request
        memory_limit = fondant_component_operation.memory_limit

        # Assign optional specification
        if cpu_request is not None:
            task.set_memory_request(cpu_request)
        if cpu_limit is not None:
            task.set_memory_limit(cpu_limit)
        if memory_request is not None:
            task.set_memory_request(memory_request)
        if memory_limit is not None:
            task.set_memory_limit(memory_limit)
        if accelerator_name is not None:
            if accelerator_name not in valid_accelerator_types:
                msg = (
                    f"Configured accelerator `{accelerator_name}` is not a valid accelerator type"
                    f"for Kubeflow compiler. Available options: {valid_accelerator_types}"
                )
                raise InvalidPipelineDefinition(msg)

            task.set_accelerator_limit(number_of_accelerators)
            if accelerator_name == "GPU":
                task.set_accelerator_type("nvidia.com/gpu")
            elif accelerator_name == "TPU":
                task.set_accelerator_type("cloud-tpus.google.com/v3")
        if node_pool_name is not None and node_pool_label is not None:
            task = self.kfp_kubernetes.add_node_selector(
                task,
                node_pool_label,
                node_pool_name,
            )
        return task


class VertexCompiler(KubeFlowCompiler):
    def __init__(self):
        super().__init__()
        self.resolve_imports()

    def resolve_imports(self):
        """Resolve imports for the Vertex compiler."""
        try:
            import kfp

            self.kfp = kfp

        except ImportError:
            msg = """You need to install kfp to use the Vertex compiler,\n
                     you can install it with `pip install fondant[vertex]`"""
            raise ImportError(
                msg,
            )

    @staticmethod
    def _set_configuration(task, fondant_component_operation):
        # Unpack optional specifications
        cpu_limit = fondant_component_operation.cpu_limit
        memory_limit = fondant_component_operation.memory_limit
        number_of_accelerators = fondant_component_operation.number_of_accelerators
        accelerator_name = fondant_component_operation.accelerator_name

        # Assign optional specification
        if cpu_limit is not None:
            task.set_cpu_limit(cpu_limit)
        if memory_limit is not None:
            task.set_memory_limit(memory_limit)
        if number_of_accelerators is not None:
            task.set_accelerator_limit(number_of_accelerators)
            if accelerator_name not in valid_vertex_accelerator_types:
                msg = (
                    f"Configured accelerator `{accelerator_name}` is not a valid accelerator type"
                    f"for Vertex compiler. Available options: {valid_vertex_accelerator_types}"
                )
                raise InvalidPipelineDefinition(msg)

            task.set_accelerator_type(accelerator_name)

        return task
