import logging
import subprocess  # nosec
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Runner(ABC):
    """Abstract base class for a runner."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Abstract method to invoke running."""


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


class KubeflowRunner(Runner):
    def __init__(self):
        self._resolve_imports()

    @abstractmethod
    def _resolve_imports(self):
        """Resolve imports for the Kubeflow compiler."""
        try:
            global kfp
            import kfp
        except ImportError:
            raise ImportError(
                "You need to install kfp to use the Kubeflow compiler, "
                / "you can install it with `pip install --extras kfp`",
            )

    def run(cls, input_spec: str, host: str, *args, **kwargs):
        """Run a kubeflow pipeline."""
        pass
        # client = kfp.Client(host=host)
        # # TODO add logic to see if pipeline exists
        # pipeline_spec = client.run_pipeline(
        #     experiment_id=experiment.id,
        #     job_name=run_name,
        #     pipeline_package_path=pipeline.package_path,
        # )

        # pipeline_url = f"{self.host}/#/runs/details/{pipeline_spec.id}"
        # logger.info(f"Pipeline is running at: {pipeline_url}")
