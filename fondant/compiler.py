import json
import logging
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)


class Compiler(ABC):
    """Abstract base class for a compiler."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @abstractmethod
    def compile(self):
        """Abstract method to invoke compilation."""


class DockerCompiler(Compiler):
    """Compiler that creates a docker-compose spec from a pipeline."""

    def compile(self, package_path: str = "docker-compose.yml") -> None:
        """Compile a pipeline to docker-compose spec that is saved on the package_path."""
        logger.info(f"Compiling {self.pipeline.name} to docker-compose.yml")
        self._patch_path()
        spec = self._generate_spec()
        with open(package_path, "w") as outfile:
            yaml.safe_dump(spec, outfile)
        logger.info(f"Successfully compiled to {package_path}")

    @staticmethod
    def _safe_component_name(component_name: str) -> str:
        """Transform a component name to a docker-compose friendly one.
        eg: `Component A` -> `component_a`.
        """
        return component_name.replace(" ", "_").lower()

    def _patch_path(self):
        """Helper that checks if the base_path is local or remote,
        if local it patches the base_path and prepares a bind mount.
        """
        base_path = Path(self.pipeline.base_path)
        # check if base path is an existing local folder
        if base_path.exists():
            self.volume = {
                "type": "bind",
                "source": str(base_path),
                "target": f"/{base_path.stem}",
            }
            self.path = f"/{base_path.stem}"
            logger.info(f"Base path set to: {self.path}")
        else:
            self.volume = None
            self.path = self.pipeline.base_path

    def _generate_spec(self) -> dict:
        """Generate a docker-compose spec as a python dictionary,
        loops over the pipeline graph to create services and their dependencies.
        """
        services = {}
        for component_name, component in self.pipeline._graph.items():
            logger.info(f"Compiling service for {component_name}")
            safe_component_name = self._safe_component_name(component_name)
            services[safe_component_name] = self.compose_service(
                component["fondant_component_op"], component["dependencies"]
            )
        return {"version": "3.8", "services": services}

    def compose_service(
        self, component_op: ComponentOp, dependencies: t.List[str]
    ) -> dict:
        """Take in a component and create a docker-compose service based on the properties."""
        # add metadata argument to command
        metadata = {"run_id": self.pipeline.name, "base_path": self.path}
        command = ["--metadata", json.dumps(metadata)]

        # add in and out manifest paths to command
        command.extend(["--output_manifest_path", f"{self.path}/manifest.txt"])

        # add arguments if any to command
        for key, value in component_op.arguments.items():
            command.extend([f"--{key}", f"{value}"])

        # resolve dependencies
        depends_on = {}
        if dependencies:
            # there is only an input manifest if the component has dependencies
            command.extend(["--input_manifest_path", f"{self.path}/manifest.txt"])
            for dependency in dependencies:
                safe_dependency = self._safe_component_name(dependency)
                depends_on[safe_dependency] = {
                    "condition": "service_completed_successfully"
                }

        volumes = [self.volume] if self.volume else []
        return {
            "image": component_op.component_spec.image,
            "command": command,
            "depends_on": depends_on,
            "volumes": volumes,
        }
