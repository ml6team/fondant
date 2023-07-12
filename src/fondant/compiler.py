import datetime
import json
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import OrderedDict

import yaml

from fondant.pipeline import Pipeline
from fondant.exceptions import InvalidPipelineExecution

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
            run_id: t.Optional[str] = None,
            resume_component: t.Optional[str] = None,
            extra_volumes: t.Optional[list] = None,
    ) -> None:
        """Compile a pipeline to docker-compose spec and save it to a specified output path.

        Args:
            pipeline: the pipeline to compile
            output_path: the path where to save the docker-compose spec
            run_id: the run_id to continue the pipeline run from
            resume_component: the name of the component to resume the run from
            extra_volumes: a list of extra volumes (using the Short syntax:
              https://docs.docker.com/compose/compose-file/05-services/#short-syntax-5)
              to mount in the docker-compose spec.
        """
        if extra_volumes is None:
            extra_volumes = []
        logger.info(f"Compiling {pipeline.name} to {output_path}")
        spec = self._generate_spec(pipeline=pipeline,
                                   extra_volumes=extra_volumes,
                                   run_id=run_id,
                                   resume_component=resume_component)
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
        return path, volume

    def _get_components_to_execute(self,
                                   *,
                                   pipeline: Pipeline,
                                   base_path: str,
                                   resume_component: t.Optional[str],
                                   run_id: t.Optional[str]) -> t.List[str]:
        """Function that returns the valid pipeline graph to resolve """

        def _get_component_execution_dict() -> t.Dict[str, bool]:
            """
            Function that returns a dictionary indicating whether a component has been executed.
            The function checks for a valid manifest file under the expected directory
            """
            execution_dict = {}
            for component_name in pipeline_graph:
                if run_id:
                    manifest_path = Path(base_path) / self._safe_component_name(
                        component_name) / run_id / "manifest.json"
                    execution_dict[component_name] = manifest_path.exists()
                else:
                    execution_dict[component_name] = False

            return execution_dict

        def _get_last_executed_components() -> t.Union[str, None]:
            """
            Function that returns the last executed component
             """
            for component_name, executed in reversed(component_execution_dict.items()):
                if executed:
                    return component_name
            else:
                return None

        def _get_components_to_execute(component_to_resume_index: int) -> t.List[str]:
            """
            Function that modifies the execution graph depending on the index of the component to
            resume the run from
            """

            if component_to_resume_index >= len(components_list):
                raise InvalidPipelineExecution(
                    f"All the components of the pipeline run `{run_id}` have been run "
                    f"successfully:\n{component_execution_dict}.\n"
                    f"You can re-run from another component by passing "
                    f"the --resume-component flag and specifying a component name.")

            component_to_execute = components_list[component_to_resume_index]
            previous_executed_component = components_list[max(component_to_resume_index - 1, 0)]

            if component_execution_dict[previous_executed_component] is False:
                raise InvalidPipelineExecution(
                    f"Cannot resume pipeline {run_id} from "
                    f"{component_to_execute}. No run was found for previous dependant component "
                    f"{previous_executed_component}.\nComponent run status: "
                    f"{component_execution_dict}")

            return components_list[component_to_resume_index:]

        pipeline_graph = OrderedDict((self._safe_component_name(component_name), component)
                                     for component_name, component in pipeline._graph.items())
        components_list = list(pipeline_graph.keys())
        component_execution_dict = _get_component_execution_dict()

        if resume_component and run_id:
            try:
                component_idx = components_list.index(resume_component)
            except ValueError:
                raise InvalidPipelineExecution \
                    (f"Specified component `{resume_component}` was not found in the list of "
                     f"pipeline components. Available components are: {components_list}")

            components_to_execute = _get_components_to_execute(component_idx)

        elif resume_component and not run_id:
            raise InvalidPipelineExecution(
                "Cannot resume from a pipeline without a specified run_id."
            )

        elif not resume_component and run_id:

            last_executed_component = _get_last_executed_components()

            if last_executed_component is not None:
                last_executed_component_idx = components_list.index(last_executed_component)
                component_to_resume_idx = last_executed_component_idx + 1
                components_to_execute = _get_components_to_execute(component_to_resume_idx)
                logger.info(f"last executed component for pipeline with run_id `{run_id}` was"
                            f" `{last_executed_component}`.\n"
                            f"Resuming run from `{components_list[component_to_resume_idx]}`"
                            f" component.")

            else:
                raise InvalidPipelineExecution(
                    f"Could not find any executed components for run with provided id "
                    f"`{run_id}` in the specified base path: `{base_path}`")

        else:
            components_to_execute = components_list

        return components_to_execute

    def _generate_spec(self,
                       pipeline: Pipeline,
                       extra_volumes: list,
                       run_id: t.Optional[str],
                       resume_component: t.Optional[str]) -> dict:
        """Generate a docker-compose spec as a python dictionary,
        loops over the pipeline graph to create services and their dependencies.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_path = pipeline.base_path
        path, volume = self._patch_path(base_path=base_path)

        components_to_execute = self._get_components_to_execute(
            pipeline=pipeline,
            base_path=base_path,
            resume_component=resume_component,
            run_id=run_id
        )

        if run_id is None:
            run_id = f"{pipeline.name}-{timestamp}"

        logger.info(f"pipeline run_id: {run_id}")
        metadata = MetaData(run_id=run_id, base_path=path)

        services = {}

        for component_name, component in pipeline._graph.items():
            logger.info(f"Compiling service for {component_name}")

            safe_component_name = self._safe_component_name(component_name)

            if safe_component_name in components_to_execute:

                component_op = component["fondant_component_op"]

                # add metadata argument to command
                command = ["--metadata", json.dumps(asdict(metadata))]

                # add in and out manifest paths to command
                command.extend(
                    [
                        "--output_manifest_path",
                        f"{path}/{safe_component_name}/{run_id}/manifest.json",
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
                # First element of docker compose always has no dependencies
                if component["dependencies"]:
                    for dependency in component["dependencies"]:
                        safe_dependency = self._safe_component_name(dependency)
                        if services:
                            depends_on[safe_dependency] = {
                                "condition": "service_completed_successfully",
                            }
                        # there is only an input manifest if the component has dependencies
                        command.extend(
                            [
                                "--input_manifest_path",
                                f"{path}/{safe_dependency}/{run_id}/manifest.json",
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
        return {
            "name": pipeline.name,
            "version": "3.8",
            "services": services,
        }
