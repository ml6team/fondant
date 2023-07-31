import datetime
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


class DockerCompiler(Compiler):
    """Compiler that creates a docker-compose spec from a pipeline."""

    def compile(
        self,
        pipeline: Pipeline,
        *,
        output_path: str = "docker-compose.yml",
        extra_volumes: t.Optional[list] = None,
        build_args: t.Optional[t.List[str]] = None,
        cache_disabled: t.Optional[bool] = True,
    ) -> None:
        """Compile a pipeline to docker-compose spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the docker-compose spec
            extra_volumes: a list of extra volumes (using the Short syntax:
              https://docs.docker.com/compose/compose-file/05-services/#short-syntax-5)
              to mount in the docker-compose spec.
            build_args: List of build arguments to pass to docker
            cache_disabled: flag to disable cached execution of components. Disabled  by default.

        """
        if extra_volumes is None:
            extra_volumes = []

        logger.info(f"Compiling {pipeline.name} to {output_path}")
        spec = self._generate_spec(
            pipeline,
            cache_disabled=cache_disabled,
            extra_volumes=extra_volumes,
            build_args=build_args or [],
        )

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        with open(output_path, "w") as outfile:
            yaml.dump(spec, outfile, Dumper=NoAliasDumper)

        logger.info(f"Successfully compiled to {output_path}")

    @staticmethod
    def _safe_component_name(component_name: str) -> str:
        """Transform a component name to a docker-compose friendly one.
        eg: `Component A` -> `component_a`.
        """
        return component_name.replace(" ", "_").lower()

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
        cache_disabled: t.Optional[bool] = True,
        extra_volumes: t.List[str],
        build_args: t.List[str],
    ) -> dict:
        """Generate a docker-compose spec as a python dictionary,
        loops over the pipeline graph to create services and their dependencies.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        path, volume = self._patch_path(base_path=pipeline.base_path)
        cache_dict = pipeline.get_pipeline_cache_dict(cache_disabled)

        services = {}

        for component_name, component in pipeline._graph.items():
            logger.info(f"Compiling service for {component_name}")
            safe_component_name = self._safe_component_name(component_name)
            cache_key = cache_dict[safe_component_name]["cache_key"]
            execute_component = cache_dict[safe_component_name]["execute_component"]
            component_op = component["fondant_component_op"]

            metadata = Metadata(
                run_id=f"{pipeline.name}-{timestamp}",
                base_path=path,
                component_id=safe_component_name,
                cache_key=cache_key,
            )

            command = [
                "--metadata",
                metadata.to_json(),
                "--output_manifest_path",
                f"{path}/{safe_component_name}/manifest_{cache_key}.json",
                "--execute_component",
                f"{execute_component}",
            ]

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
                    cache_key = cache_dict[safe_dependency]["cache_key"]
                    command.extend(
                        [
                            "--input_manifest_path",
                            f"{path}/{safe_dependency}/manifest_{cache_key}.json",
                        ],
                    )

            volumes: t.List[t.Union[str, dict]] = []
            if volume:
                volumes.append(asdict(volume))
            if extra_volumes:
                volumes.extend(extra_volumes)

            services[safe_component_name] = {
                "command": command,
                "depends_on": depends_on,
                "volumes": volumes,
            }

            if component_op.dockerfile_path is not None:
                logger.info(
                    f"Found Dockerfile for {component_name}, adding build step.",
                )
                services[safe_component_name]["build"] = {
                    "context": str(component_op.component_dir),
                    "args": build_args,
                }
            else:
                services[safe_component_name][
                    "image"
                ] = component_op.component_spec.image
        return {
            "name": pipeline.name,
            "version": "3.8",
            "services": services,
        }
