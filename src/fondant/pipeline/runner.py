import logging
import os
import subprocess  # nosec
import sys
import typing as t
from abc import ABC, abstractmethod

import yaml

from fondant.core.schema import CloudCredentialsMount
from fondant.pipeline import Pipeline
from fondant.pipeline.compiler import (
    DockerCompiler,
    KubeFlowCompiler,
    SagemakerCompiler,
    VertexCompiler,
)

logger = logging.getLogger(__name__)


class Runner(ABC):
    """Abstract base class for a runner."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Abstract method to invoke running."""


class DockerRunner(Runner):
    def _run(self, input_spec: str, *args, **kwargs):
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

        print("Starting pipeline run...")

        # copy the current environment with the DOCKER_DEFAULT_PLATFORM argument
        subprocess.call(  # nosec
            cmd,
            env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
        )
        print("Finished pipeline run.")

    def run(
        self,
        input: t.Union[Pipeline, str],
        *,
        extra_volumes: t.Union[t.Optional[list], t.Optional[str]] = None,
        build_args: t.Optional[t.List[str]] = None,
        auth_provider: t.Optional[CloudCredentialsMount] = None,
    ) -> None:
        """Run a pipeline, either from a compiled docker-compose spec or from a fondant pipeline.

        Args:
            input: the pipeline to compile or a path to a already compiled docker-compose spec
            extra_volumes: a list of extra volumes (using the Short syntax:
             https://docs.docker.com/compose/compose-file/05-services/#short-syntax-5)
             to mount in the docker-compose spec.
            build_args: List of build arguments to pass to docker
            auth_provider: The cloud provider to use for authentication. Default is None.
        """
        self.check_docker_install()
        self.check_docker_compose_install()

        if isinstance(input, Pipeline):
            os.makedirs(".fondant", exist_ok=True)
            output_path = ".fondant/compose.yaml"
            logging.info(
                "Found reference to un-compiled pipeline... compiling",
            )
            compiler = DockerCompiler()
            compiler.compile(
                input,
                output_path=output_path,
                extra_volumes=extra_volumes,
                build_args=build_args,
                auth_provider=auth_provider,
            )
            self._run(output_path)
        else:
            self._run(input)

    @staticmethod
    def check_docker_install():
        """Execute docker command to check if docker is available."""
        try:
            # Check Docker info
            subprocess.check_call(  # nosec
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
            )

        except subprocess.CalledProcessError:
            sys.exit(
                "Docker is not installed or not running. Please make sure "
                "Docker is installed and is running."
                "Find more details on the Docker installation here: "
                "https://fondant.ai/en/latest/guides/installation/#docker-installation",
            )

    @staticmethod
    def check_docker_compose_install():
        """Execute docker compose command to check if docker is available."""
        try:
            # Check Docker info
            subprocess.check_call(  # nosec
                ["docker", "compose", "version"],
                stdout=subprocess.DEVNULL,
            )

        except subprocess.CalledProcessError:
            sys.exit(
                "Docker Compose is not installed or not running. Please make sure "
                "Docker Compose is installed."
                "Find more details on the Docker installation here: "
                "https://fondant.ai/en/latest/guides/installation/#docker-installation",
            )


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
        input: t.Union[Pipeline, str],
        *,
        experiment_name: str = "Default",
    ):
        """Run a pipeline, either from a compiled kubeflow spec or from a fondant pipeline.

        Args:
            input: the pipeline to compile or a path to a already compiled sagemaker spec
            experiment_name: the name of the experiment to create
        """
        if isinstance(input, Pipeline):
            os.makedirs(".fondant", exist_ok=True)
            output_path = ".fondant/kubeflow-pipeline.yaml"
            logging.info(
                "Found reference to un-compiled pipeline... compiling",
            )
            compiler = KubeFlowCompiler()
            compiler.compile(
                input,
                output_path=output_path,
            )
            self._run(output_path, experiment_name=experiment_name)
        else:
            self._run(input, experiment_name=experiment_name)

    def _run(
        self,
        input_spec: str,
        *,
        experiment_name: str = "Default",
    ):
        """Run a kubeflow pipeline."""
        try:
            experiment = self.client.get_experiment(experiment_name=experiment_name)
        except ValueError:
            logger.info(
                f"defined experiment '{experiment_name}' not found. creating new experiment"
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

    def run(
        self,
        input: t.Union[Pipeline, str],
    ):
        """Run a pipeline, either from a compiled vertex spec or from a fondant pipeline.

        Args:
            input: the pipeline to compile or a path to a already compiled sagemaker spec
        """
        if isinstance(input, Pipeline):
            os.makedirs(".fondant", exist_ok=True)
            output_path = ".fondant/vertex-pipeline.yaml"
            logging.info(
                "Found reference to un-compiled pipeline... compiling",
            )
            compiler = VertexCompiler()
            compiler.compile(
                input,
                output_path=output_path,
            )
            self._run(output_path)
        else:
            self._run(input)

    def _run(self, input_spec: str, *args, **kwargs):
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


class SagemakerRunner(Runner):
    def __init__(self):
        self.__resolve_imports()
        self.client = self.boto3.client("sagemaker")

    def __resolve_imports(self):
        try:
            import boto3

            self.boto3 = boto3
        except ImportError:
            msg = (
                """You need to install boto3 to use the Sagemaker compiler,\n
                     you can install it with `pip install fondant[sagemaker]`""",
            )
            raise ImportError(msg)

    def run(
        self,
        input: t.Union[Pipeline, str],
        pipeline_name: str,
        role_arn: str,
    ):
        """Run a pipeline, either from a compiled sagemaker spec or from a fondant pipeline.

        Args:
            input: the pipeline to compile or a path to a already compiled sagemaker spec
            pipeline_name: the name of the pipeline to create
            role_arn: the Amazon Resource Name role to use for the processing steps,
            if none provided the `sagemaker.get_execution_role()` role will be used.
        """
        if isinstance(input, Pipeline):
            os.makedirs(".fondant", exist_ok=True)
            output_path = ".fondant/sagemaker-pipeline.yaml"
            logging.info(
                "Found reference to un-compiled pipeline... compiling",
            )
            compiler = SagemakerCompiler()
            compiler.compile(
                input,
                output_path=output_path,
                role_arn=role_arn,
            )
            self._run(output_path, pipeline_name=pipeline_name, role_arn=role_arn)
        else:
            self._run(input, pipeline_name=pipeline_name, role_arn=role_arn)

    def _run(self, input_spec: str, pipeline_name: str, role_arn: str):
        """Creates/updates a sagemaker pipeline and execute it."""
        with open(input_spec) as f:
            pipeline = f.read()
            pipelines = self.client.list_pipelines(
                PipelineNamePrefix=pipeline_name,
            )
            if pipelines["PipelineSummaries"]:
                logging.info(
                    f"Pipeline with name {pipeline_name} already exists, updating it",
                )
                _ = self.client.update_pipeline(
                    PipelineName=pipeline_name,
                    PipelineDefinition=pipeline,
                    RoleArn=role_arn,
                )
            else:
                logging.info(
                    f"Pipeline with name {pipeline_name} does not exist, creating it",
                )
                _ = self.client.create_pipeline(
                    PipelineName=pipeline_name,
                    PipelineDefinition=pipeline,
                    RoleArn=role_arn,
                )

        logging.info(f"Starting pipeline execution for pipeline {pipeline_name}")
        _ = self.client.start_pipeline_execution(
            PipelineName=pipeline_name,
            ParallelismConfiguration={"MaxParallelExecutionSteps": 1},
        )
        logging.info(
            "Pipeline execution started for pipeline, visit Sagemaker studio to follow up",
        )
