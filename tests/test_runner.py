import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from fondant.runner import DockerRunner, KubeflowRunner

VALID_PIPELINE = Path("./tests/example_pipelines/compiled_pipeline/")


def test_docker_runner():
    """Test that the docker runner while mocking subprocess.call."""
    with mock.patch("subprocess.call") as mock_call:
        DockerRunner().run("some/path")
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
        )


class MockKfpClient:
    def __init__(self, host):
        self.host = host
        self._experiments = {"Default": SimpleNamespace(id="123")}

    def get_experiment(self, experiment_name):
        try:
            return self._experiments[experiment_name]
        except KeyError:
            raise ValueError

    def create_experiment(self, experiment_name):
        self._experiments[experiment_name] = SimpleNamespace(id="456")
        return self.get_experiment(experiment_name)

    def run_pipeline(self, experiment_id, job_name, pipeline_package_path):
        return SimpleNamespace(id="xyz")


def test_kubeflow_runner():
    input_spec_path = str(VALID_PIPELINE / "kubeflow_pipeline.yml")
    with mock.patch(
        "kfp.Client",
        new=MockKfpClient,
    ):
        runner = KubeflowRunner(host="some_host")
        runner.run(input_spec=input_spec_path)

        assert runner.client.host == "some_host"


def test_kubeflow_runner_new_experiment():
    input_spec_path = str(VALID_PIPELINE / "kubeflow_pipeline.yml")
    with mock.patch(
        "kfp.Client",
        new=MockKfpClient,
    ):
        runner = KubeflowRunner(host="some_host")
        runner.run(input_spec=input_spec_path, experiment_name="NewExperiment")


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
