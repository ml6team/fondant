from unittest.mock import patch

from fondant.runner import DockerRunner


def test_docker_runner():
    """Test that the docker runner while mocking subprocess.call."""
    with patch("subprocess.call") as mock_call:
        DockerRunner().run("some/path")
        mock_call.assert_called_once_with(
            ["docker", "compose", "-f", "some/path", "up", "--build"],
        )
