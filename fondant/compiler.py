import datetime
import json
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from fondant.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Compiler(ABC):
    """Abstract base class for a compiler."""

    @abstractmethod
    def compile(self, *args, **kwargs):
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


@dataclass
class MetaData:
    """Dataclass representing the metadata arguments of a pipeline.

    Args:
        run_id: identifier of the current pipeline run
        base_path: the base path used to store the artifacts.
    """

    run_id: str
    base_path: str


class DockerCompiler(Compiler):
    """Compiler that creates a docker-compose spec from a pipeline."""

    def compile(
        self,
        pipeline: Pipeline,
        output_path: str = "docker-compose.yml",
        extra_volumes: t.Optional[list] = None,
    ) -> None:
        """Compile a pipeline to docker-compose spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the docker-compose spec
            extra_volumes: a list of extra volumes (using the Short syntax:
              https://docs.docker.com/compose/compose-file/05-services/#short-syntax-5)
              to mount in the docker-compose spec.
        """
        if extra_volumes is None:
            extra_volumes = []

        logger.info(f"Compiling {pipeline.name} to {output_path}")
        spec = self._generate_spec(pipeline=pipeline, extra_volumes=extra_volumes)
        with open(output_path, "w") as outfile:
            yaml.safe_dump(spec, outfile)
        logger.info(f"Successfully compiled to {output_path}")

    @staticmethod
    def _safe_component_name(component_name: str) -> str:
        """Transform a component name to a docker-compose friendly one.
        eg: `Component A` -> `component_a`.
        """
        return component_name.replace(" ", "_").lower()

    def _patch_path(self, base_path: str) -> t.Tuple[str, t.Optional[DockerVolume]]:
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
        return (path, volume)

    def _generate_spec(self, pipeline: Pipeline, extra_volumes: list) -> dict:
        """Generate a docker-compose spec as a python dictionary,
        loops over the pipeline graph to create services and their dependencies.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        path, volume = self._patch_path(base_path=pipeline.base_path)
        metadata = MetaData(run_id=f"{pipeline.name}-{timestamp}", base_path=path)

        services = {}

        for component_name, component in pipeline._graph.items():
            logger.info(f"Compiling service for {component_name}")
            safe_component_name = self._safe_component_name(component_name)

            component_op = component["fondant_component_op"]

            # add metadata argument to command
            command = ["--metadata", json.dumps(asdict(metadata))]

            # add in and out manifest paths to command
            command.extend(
                [
                    "--output_manifest_path",
                    f"{path}/{safe_component_name}/manifest.json",
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
                    safe_dependency = self._safe_component_name(dependency)
                    depends_on[safe_dependency] = {
                        "condition": "service_completed_successfully",
                    }
                    # there is only an input manifest if the component has dependencies
                    command.extend(
                        [
                            "--input_manifest_path",
                            f"{path}/{safe_dependency}/manifest.json",
                        ],
                    )

            volumes = []
            if volume:
                volumes.append(asdict(volume))
            if extra_volumes:
                volumes.extend(extra_volumes)

            services[safe_component_name] = {
                "command": command,
                "depends_on": depends_on,
                "volumes": volumes,
            }

            if component_op.local_component:
                services[safe_component_name][
                    "build"
                ] = f"./{Path(component_op.component_spec_path).parent}"
            else:
                services[safe_component_name][
                    "image"
                ] = component_op.component_spec.image
        return {"version": "3.8", "services": services}
