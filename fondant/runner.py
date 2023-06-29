import subprocess  # nosec
from abc import ABC, abstractmethod


class Runner(ABC):
    """Abstract base class for a runner."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Abstract method to invoke running."""


class DockerRunner(Runner):
    def run(cls, input_spec: str, *args, **kwargs):
        """Run a docker-compose spec."""
        cmd = ["docker", "compose", "-f", input_spec, "up", "--build"]

        subprocess.call(cmd)  # nosec
