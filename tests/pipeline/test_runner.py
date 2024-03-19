import os
import sys
from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace
from unittest import mock

import pytest
from fondant.dataset import Dataset, Workspace
from fondant.dataset.runner import (
    DockerRunner,
    KubeflowRunner,
    SagemakerRunner,
    VertexRunner,
)

VALID_PIPELINE = Path("./tests/pipeline/examples/pipelines/compiled_pipeline/")

WORKSPACE = Workspace(
    name="testpipeline",
    description="description of the test pipeline",
    base_path="/foo/bar",
)


@pytest.fixture()
def mock_docker_installation(monkeypatch):  # noqa: PT004
    def mock_check_docker_install(self):
        pass

    def mock_check_docker_compose_install(self):
        pass

    monkeypatch.setattr(DockerRunner, "check_docker_install", mock_check_docker_install)
    monkeypatch.setattr(
        DockerRunner,
        "check_docker_compose_install",
        mock_check_docker_compose_install,
    )


def test_docker_runner(mock_docker_installation):
    """Test that the docker runner while mocking subprocess.call."""
    with mock.patch("subprocess.call") as mock_call:
        DockerRunner().run("some/path", workspace=WORKSPACE)
        mock_call.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                "some/path",
                "up",
                "--build",
                "--pull",
                "always",
                "--remove-orphans",
            ],
            env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
        )


def test_docker_runner_from_pipeline(mock_docker_installation, tmp_path_factory):
    with mock.patch("subprocess.call") as mock_call, tmp_path_factory.mktemp(
        "temp",
    ) as fn:
        workspace = Workspace(
            name="testpipeline",
            description="description of the test pipeline",
            base_path=str(fn),
        )

        dataset = Dataset()
        DockerRunner().run(dataset, workspace)
        mock_call.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                ".fondant/compose.yaml",
                "up",
                "--build",
                "--pull",
                "always",
                "--remove-orphans",
            ],
            env=dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64"),
        )


def test_docker_is_not_available():
    expected_msg = (
        "Docker is not installed or not running. Please make sure Docker is installed and is "
        "running.Find more details on the Docker installation here: "
        "https://fondant.ai/en/latest/guides/installation/#docker-installation"
    )
    with mock.patch(
        "subprocess.check_output",
        side_effect=CalledProcessError(returncode=1, cmd=""),
    ), pytest.raises(SystemExit, match=str(expected_msg)):
        DockerRunner().check_docker_install()


def test_docker_version_is_not_supported():
    expected_msg = (
        "Docker version is not compatible. Please make sure "
        "You have Docker version 20.10.0 or higher installed. "
        "Your current version is: "
    )
    with mock.patch(
        "subprocess.check_output",
        return_value=b"1.3.8",
    ), pytest.raises(SystemExit, match=expected_msg):
        DockerRunner().check_docker_install()


def test_docker_compose_is_not_available():
    expected_msg = (
        "Docker Compose is not installed or not running. Please make sure Docker Compose is "
        "installed.Find more details on the Docker installation here: "
        "https://fondant.ai/en/latest/guides/installation/#docker-installation"
    )

    with mock.patch(
        "subprocess.check_output",
        side_effect=CalledProcessError(returncode=1, cmd=""),
    ), pytest.raises(SystemExit, match=expected_msg):
        DockerRunner().check_docker_compose_install()


def test_docker_compose_version_is_not_supported():
    expected_msg = (
        "Docker Compose version is not compatible. Please make sure "
        "You have Docker Compose version 2.20.0 or higher installed. "
        "Your current version is: "
    )
    with mock.patch(
        "subprocess.check_output",
        return_value=b"1.24.5-desktop.1",
    ), pytest.raises(SystemExit, match=expected_msg):
        DockerRunner().check_docker_compose_install()


class MockKfpClient:
    def __init__(self, host):
        self.host = host
        self._experiments = {"Default": SimpleNamespace(experiment_id="123")}

    def get_experiment(self, experiment_name):
        try:
            return self._experiments[experiment_name]
        except KeyError:
            raise ValueError

    def create_experiment(self, experiment_name):
        self._experiments[experiment_name] = SimpleNamespace(experiment_id="456")
        return self.get_experiment(experiment_name)

    def run_pipeline(self, experiment_id, job_name, pipeline_package_path):
        return SimpleNamespace(run_id="xyz")


def test_kubeflow_runner():
    input_spec_path = str(VALID_PIPELINE / "kubeflow_pipeline.yml")
    with mock.patch(
        "kfp.Client",
        new=MockKfpClient,
    ):
        runner = KubeflowRunner(host="some_host")
        runner.run(dataset=input_spec_path, workspace=WORKSPACE)

        assert runner.client.host == "some_host"


def test_kubeflow_runner_new_experiment():
    input_spec_path = str(VALID_PIPELINE / "kubeflow_pipeline.yml")
    with mock.patch(
        "kfp.Client",
        new=MockKfpClient,
    ):
        runner = KubeflowRunner(host="some_host")
        runner.run(
            dataset=input_spec_path,
            experiment_name="NewExperiment",
            workspace=WORKSPACE,
        )


def test_kfp_import():
    """Test that the kfp import throws the correct error."""
    with mock.patch.dict(sys.modules), mock.patch(
        "kfp.Client",
        new=MockKfpClient,
    ):
        # remove kfp from the modules
        sys.modules["kfp"] = None
        with pytest.raises(ImportError):
            _ = KubeflowRunner(host="some_host")


class MockKubeFlowCompiler:
    def compile(
        self,
        dataset,
        output_path,
    ) -> None:
        with open(output_path, "w") as f:
            f.write("foo: bar")


def test_kubeflow_runner_from_pipeline():
    with mock.patch(
        "fondant.dataset.runner.KubeFlowCompiler",
        new=MockKubeFlowCompiler,
    ), mock.patch(
        "fondant.dataset.runner.KubeflowRunner._run",
    ) as mock_run, mock.patch(
        "kfp.Client",
        new=MockKfpClient,
    ):
        runner = KubeflowRunner(host="some_host")
        dataset = Dataset()
        runner.run(
            dataset=dataset,
            workspace=WORKSPACE,
        )

        mock_run.assert_called_once_with(
            ".fondant/kubeflow-pipeline.yaml",
            experiment_name="Default",
        )


def test_vertex_runner():
    input_spec_path = str(VALID_PIPELINE / "kubeflow_pipeline.yml")
    with mock.patch("google.cloud.aiplatform.init", return_value=None), mock.patch(
        "google.cloud.aiplatform.PipelineJob",
    ):
        runner = VertexRunner(project_id="some_project", region="some_region")
        runner.run(input=input_spec_path, workspace=WORKSPACE)

        # test with service account
        runner2 = VertexRunner(
            project_id="some_project",
            region="some_region",
            service_account="some_account",
        )
        runner2.run(input=input_spec_path, workspace=WORKSPACE)


def test_vertex_runner_from_pipeline():
    with mock.patch(
        "fondant.dataset.runner.VertexCompiler",
        new=MockKubeFlowCompiler,
    ), mock.patch("fondant.dataset.runner.VertexRunner._run") as mock_run, mock.patch(
        "google.cloud.aiplatform.init",
        return_value=None,
    ):
        runner = VertexRunner(project_id="some_project", region="some_region")
        runner.run(
            input=Dataset(),
            workspace=WORKSPACE,
        )

        mock_run.assert_called_once_with(".fondant/vertex-pipeline.yaml")


def test_sagemaker_runner(tmp_path_factory):
    with mock.patch("boto3.client", spec=True), tmp_path_factory.mktemp(
        "temp",
    ) as tmpdir:
        # create a small temporary spec file
        with open(tmpdir / "spec.json", "w") as f:
            f.write('{"pipelineInfo": {"name": "pipeline_1"}}')
        runner = SagemakerRunner()

        runner.run(
            dataset=tmpdir / "spec.json",
            workspace=WORKSPACE,
            pipeline_name="pipeline_1",
            role_arn="arn:something",
        )

        # check which methods were called on the client
        assert runner.client.method_calls == [
            mock.call.list_pipelines(PipelineNamePrefix="pipeline_1"),
            mock.call.update_pipeline(
                PipelineName="pipeline_1",
                PipelineDefinition='{"pipelineInfo": {"name": "pipeline_1"}}',
                RoleArn="arn:something",
            ),
            mock.call.start_pipeline_execution(
                PipelineName="pipeline_1",
                ParallelismConfiguration={"MaxParallelExecutionSteps": 1},
            ),
        ]

        # reset the mock and test the creation of a new pipeline
        runner.client.reset_mock()
        runner.client.configure_mock(
            **{"list_pipelines.return_value": {"PipelineSummaries": []}},
        )

        runner.run(
            dataset=tmpdir / "spec.json",
            workspace=WORKSPACE,
            pipeline_name="pipeline_1",
            role_arn="arn:something",
        )
        # here we expect the create_pipeline method to be called
        assert runner.client.method_calls == [
            mock.call.list_pipelines(PipelineNamePrefix="pipeline_1"),
            mock.call.create_pipeline(
                PipelineName="pipeline_1",
                PipelineDefinition='{"pipelineInfo": {"name": "pipeline_1"}}',
                RoleArn="arn:something",
            ),
            mock.call.start_pipeline_execution(
                PipelineName="pipeline_1",
                ParallelismConfiguration={"MaxParallelExecutionSteps": 1},
            ),
        ]


class MockSagemakerCompiler:
    def compile(
        self,
        pipeline,
        output_path,
        *,
        role_arn,
    ) -> None:
        with open(output_path, "w") as f:
            f.write("foo: bar")


def test_sagemaker_runner_from_pipeline():
    with mock.patch(
        "fondant.dataset.runner.SagemakerCompiler",
        new=MockSagemakerCompiler,
    ), mock.patch("boto3.client", spec=True):
        runner = SagemakerRunner()
        runner.run(
            dataset=Dataset(),
            workspace=WORKSPACE,
            pipeline_name=WORKSPACE.name,
            role_arn="arn:something",
        )
