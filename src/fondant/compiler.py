import json
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from fondant.manifest import Metadata
from fondant.pipeline import Pipeline

logger = logging.getLogger(__name__)

DASK_DIAGNOSTIC_DASHBOARD_PORT = 8787


class Compiler(ABC):
    """Abstract base class for a compiler."""

    @abstractmethod
    def compile(self, *args, **kwargs) -> None:
        """Abstract method to invoke compilation."""


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

        for component_name, component in pipeline._graph.items():
            component_op = component["fondant_component_op"]

            metadata = Metadata(
                pipeline_name=pipeline.name,
                run_id=run_id,
                base_path=path,
                component_id=component_name,
                cache_key=component_op.get_component_cache_key(),
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

            if component_op.number_of_gpus is not None:
                services[component_name]["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": component_op.number_of_gpus,
                                    "capabilities": ["gpu"],
                                },
                            ],
                        },
                    },
                }

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


class KubeFlowCompiler(Compiler):
    """Compiler that creates a Kubeflow pipeline spec from a pipeline."""

    def __init__(self):
        self._resolve_imports()

    def _resolve_imports(self):
        """Resolve imports for the Kubeflow compiler."""
        try:
            import kfp
            import kfp.gcp as kfp_gcp

            self.kfp = kfp
            self.kfp_gcp = kfp_gcp
        except ImportError:
            msg = """You need to install kfp to use the Kubeflow compiler,\n
                     you can install it with `pip install fondant[kfp]`"""
            raise ImportError(
                msg,
            )

    def compile(
        self,
        pipeline: Pipeline,
        output_path: str = "kubeflow_pipeline.yml",
    ) -> None:
        """Compile a pipeline to Kubeflow pipeline spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the Kubeflow pipeline spec
        """
        run_id = pipeline.get_run_id()

        @self.kfp.dsl.pipeline(name=pipeline.name, description=pipeline.description)
        def kfp_pipeline():
            previous_component_task = None
            manifest_path = ""

            for component_name, component in pipeline._graph.items():
                component_op = component["fondant_component_op"]

                metadata = Metadata(
                    pipeline_name=pipeline.name,
                    run_id=run_id,
                    base_path=pipeline.base_path,
                    component_id=component_name,
                    cache_key=component_op.get_component_cache_key(),
                )

                logger.info(f"Compiling service for {component_name}")

                # convert ComponentOp to Kubeflow component
                kubeflow_component_op = self.kfp.components.load_component(
                    text=component_op.component_spec.kubeflow_specification.to_string(),
                )

                # Execute the Kubeflow component and pass in the output manifest path from
                # the previous component.
                component_args = component_op.arguments

                component_task = kubeflow_component_op(
                    input_manifest_path=manifest_path,
                    metadata=metadata.to_json(),
                    **component_args,
                )
                # Set optional configurations
                component_task = self._set_configuration(
                    component_task,
                    component_op,
                )

                # Set image pull policy to always
                component_task.container.set_image_pull_policy("Always")

                # Set the execution order of the component task to be after the previous
                # component task.
                if previous_component_task is not None:
                    component_task.after(previous_component_task)

                # Update the manifest path to be the output path of the current component task.
                manifest_path = component_task.outputs["output_manifest_path"]

                previous_component_task = component_task

        self.pipeline = pipeline
        self.pipeline.validate(run_id=run_id)
        logger.info(f"Compiling {self.pipeline.name} to {output_path}")

        self.kfp.compiler.Compiler().compile(kfp_pipeline, output_path)  # type: ignore
        logger.info("Pipeline compiled successfully")

    def _set_configuration(self, task, fondant_component_operation):
        # Unpack optional specifications
        number_of_gpus = fondant_component_operation.number_of_gpus
        node_pool_label = fondant_component_operation.node_pool_label
        node_pool_name = fondant_component_operation.node_pool_name
        preemptible = fondant_component_operation.preemptible

        # Assign optional specification
        if number_of_gpus is not None:
            task.set_gpu_limit(number_of_gpus)
        if node_pool_name is not None and node_pool_label is not None:
            task.add_node_selector_constraint(node_pool_label, node_pool_name)
        if preemptible is True:
            logger.warning(
                f"Preemptible VM enabled on component `{fondant_component_operation.name}`. Please"
                f" note that Preemptible nodepools only works on clusters setup on GCP and "
                f"with nodepools pre-configured with preemptible VMs. More info here:"
                f" https://v1-6-branch.kubeflow.org/docs/distributions/gke/pipelines/preemptible/",
            )
            task.apply(self.kfp_gcp.use_preemptible_nodepool())

        return task
