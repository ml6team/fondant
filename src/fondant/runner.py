import json
import logging
import subprocess  # nosec
from abc import ABC, abstractmethod

import yaml

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
    def __init__(self, host: str):
        self._resolve_imports()
        self.host = host
        self.client = self.kfp.Client(host=host)

    def _resolve_imports(self):
        """Resolve imports for the Kubeflow compiler."""
        try:
            import kfp

            self.kfp = kfp
        except ImportError:
            msg = """You need to install kfp to use the Kubeflow compiler,\n
                     you can install it with `pip install fondant[kfp]`"""
            raise ImportError(
                msg,
            )

    def run(
        self,
        input_spec: str,
        experiment_name: str = "Default",
        *args,
        **kwargs,
    ):
        """Run a kubeflow pipeline."""
        try:
            experiment = self.client.get_experiment(experiment_name=experiment_name)
        except ValueError:
            logger.info(
                f"Defined experiment '{experiment_name}' not found. Creating new experiment"
                f" under this name",
            )
            experiment = self.client.create_experiment(experiment_name)

        job_name = self.get_name_from_spec(input_spec) + "_run"
        # TODO add logic to see if pipeline exists
        runner = self.client.run_pipeline(
            experiment_id=experiment.id,
            job_name=job_name,
            pipeline_package_path=input_spec,
        )

        pipeline_url = f"{self.host}/#/runs/details/{runner.id}"
        logger.info(f"Pipeline is running at: {pipeline_url}")

    def get_name_from_spec(self, input_spec: str):
        """Get the name of the pipeline from the spec."""
        with open(input_spec) as f:
            spec = yaml.safe_load(f)
            return json.loads(
                spec["metadata"]["annotations"]["pipelines.kubeflow.org/pipeline_spec"],
            )["name"]
