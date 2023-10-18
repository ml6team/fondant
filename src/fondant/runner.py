import logging
import subprocess  # nosec
import typing as t
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
            experiment_id=experiment.experiment_id,
            job_name=job_name,
            pipeline_package_path=input_spec,
        )

        pipeline_url = f"{self.host}/#/runs/details/{runner.run_id}"
        logger.info(f"Pipeline is running at: {pipeline_url}")

    def get_name_from_spec(self, input_spec: str):
        """Get the name of the pipeline from the spec."""
        with open(input_spec) as f:
            spec, *_ = yaml.safe_load_all(f)
            return spec["pipelineInfo"]["name"]


class VertexRunner(Runner):
    def __resolve_imports(self):
        import google.cloud.aiplatform as aip

        self.aip = aip

    def __init__(
        self,
        project_id: str,
        region: str,
        service_account: t.Optional[str] = None,
        network: t.Optional[str] = None,
    ):
        self.__resolve_imports()

        self.aip.init(
            project=project_id,
            location=region,
        )
        self.service_account = service_account
        self.network = network

    def run(self, input_spec: str, *args, **kwargs):
        job = self.aip.PipelineJob(
            display_name=self.get_name_from_spec(input_spec),
            template_path=input_spec,
            enable_caching=False,
        )
        job.submit(
            service_account=self.service_account,
            network=self.network,
        )

    def get_name_from_spec(self, input_spec: str):
        """Get the name of the pipeline from the spec."""
        with open(input_spec) as f:
            spec = yaml.safe_load(f)
            return spec["pipelineInfo"]["name"]
