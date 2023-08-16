import logging
import subprocess  # nosec
import typing as t
from abc import ABC, abstractmethod

from fondant.component import Component
from fondant.executor import (
    DaskLoadExecutor,
    DaskTransformExecutor,
    DaskWriteExecutor,
    Executor,
    PandasTransformExecutor,
)

logger = logging.getLogger(__name__)

COMPONENT_EXECUTOR_MAPPER: t.Dict[str, t.Type[Executor]] = {
    "DaskLoadComponent": DaskLoadExecutor,
    "DaskTransformComponent": DaskTransformExecutor,
    "DaskWriteComponent": DaskWriteExecutor,
    "PandasTransformExecutor": PandasTransformExecutor,
}


class Runner(ABC):
    """Abstract base class for a runner."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Abstract method to invoke running."""


class ComponentRunner(Runner):
    def __init__(self, component: t.Type[Component]):
        self.component = component

    def _get_executor(self) -> Executor:
        component_type = self.component.__bases__[0].__name__
        return COMPONENT_EXECUTOR_MAPPER[component_type].from_args()

    def run(self):
        executor = self._get_executor()
        executor.execute(self.component)


class DockerRunner(Runner):
    def run(self, input_spec: str, *args, **kwargs):
        """Run a docker-compose spec."""
        cmd = [
            "docker",
            "compose",
            "-f",
            input_spec,
            "up",
            "--build",
            "--pull",
            "always",
            "--remove-orphans",
        ]

        subprocess.call(cmd)  # nosec
