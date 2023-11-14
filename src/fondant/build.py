"""Module holding implementation to build Fondant components, used by the `fondant build`
command.
"""
import logging
import re
import typing as t
from pathlib import Path

from fondant.pipeline import ComponentOp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_component(  # ruff: noqa: PLR0912, PLR0915
    component_dir: Path,
    *,
    tag: t.Optional[str],
    build_args: t.List[str],
    labels: t.List[str],
    nocache: bool = False,
    pull: bool = False,
    target: t.Optional[str] = None,
) -> None:
    try:
        import docker
    except ImportError:
        msg = (
            "You need to install `docker` to use the `fondant build` command, you can install "
            "it with `pip install fondant[docker]`"
        )
        raise SystemExit(
            msg,
        )

    component_op = ComponentOp(component_dir)
    component_spec = component_op.component_spec

    if component_op.dockerfile_path is None:
        msg = (
            f"Could not detect a `Dockerfile` in {component_dir}. Please make sure it is placed "
            f"at the root of your component_dir and named `Dockerfile`."
        )
        raise SystemExit(msg)

    if tag is None:
        logger.info("No tag provided. Extracting image name from `component_spec.yaml`")
        full_image_name = component_spec.image
    elif ":" in tag:
        logger.info("Detected `:` in tag")
        full_image_name = tag
    else:
        logger.info("Did not detect `:` in tag")
        logger.info("Extracting image name from `component_spec.yaml`")
        repository = component_spec.image.split(":")[0]
        full_image_name = f"{repository}:{tag}"

    logger.info(f"Assuming full image name: {full_image_name}")

    logger.info("Building image...")

    # Convert build args from ["key=value", ...] to {"key": "value", ...}
    build_kwargs = {}
    for arg in build_args:
        k, v = arg.split("=", 1)
        build_kwargs[k] = v

    # Convert label args from ["key=value", ...] to {"key": "value", ...}
    label_kwargs = {}
    for arg in labels:
        k, v = arg.split("=", 1)
        label_kwargs[k] = v

    try:
        docker_client = docker.from_env()
    except docker.errors.DockerException:
        for url in [
            "/var/run/docker.sock",
            Path.home() / ".docker/desktop/docker.sock",
            Path.home() / ".docker/run/docker.sock",
        ]:
            base_url = f"unix://{url}"
            try:
                docker_client = docker.DockerClient(base_url=base_url)
                break
            except docker.errors.DockerException:
                continue
        else:
            msg = "Could not connect to docker daemon, is it running?"
            raise SystemExit(msg)

    logs = docker_client.api.build(
        path=str(component_dir),
        tag=full_image_name,
        buildargs=build_kwargs,
        nocache=nocache,
        pull=pull,
        target=target,
        decode=True,
        labels=label_kwargs,
    )

    for chunk in logs:
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                logger.info(line)

    logger.info("Pushing image...")
    repository, tag = full_image_name.split(":")
    logs = docker_client.api.push(repository, tag=tag, stream=True, decode=True)

    for chunk in logs:
        if "error" in chunk:
            logger.error("Push failed:")
            raise SystemExit(chunk["error"])
        message = chunk.get("status", "")
        if "progress" in chunk:
            message += " | " + chunk["progress"]
        logger.info(message)

    logger.info("Updating image name in component_spec")
    # Read and write with `re.sub` to prevent reformatting of file with yaml
    with open(component_dir / component_op.COMPONENT_SPEC_NAME, "r+") as f:
        content = f.read()
        f.seek(0)
        content = re.sub(r"image: [^\n]*", f"image: {full_image_name}", content)
        f.write(content)
        f.truncate()

    logger.info("Done")
