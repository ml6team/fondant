"""Fondant pipelines test."""
from pathlib import Path

import pytest

from fondant.exceptions import InvalidPipelineDefinition
from fondant.pipeline import FondantComponentOp, FondantPipeline

valid_pipeline_path = Path(__file__).parent / "example_pipelines/valid_pipeline"
invalid_pipeline_path = Path(__file__).parent / "example_pipelines/invalid_pipeline"


@pytest.fixture
def default_pipeline_args():
    return {
        "pipeline_name": "pipeline",
        "base_path": "gcs://bucket/blob",
    }


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            ["first_component.yaml", "second_component.yaml", "third_component.yaml"],
        ),
    ],
)
def test_valid_pipeline(default_pipeline_args, valid_pipeline_example, tmp_path):
    """Test that a valid pipeline definition can be compiled without errors."""
    example_dir, component_names = valid_pipeline_example
    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(valid_pipeline_path / example_dir)

    pipeline = FondantPipeline(**default_pipeline_args)

    first_component_op = FondantComponentOp(
        Path(components_path / component_names[0]), component_args
    )
    second_component_op = FondantComponentOp(
        Path(components_path / component_names[1]), component_args
    )
    third_component_op = FondantComponentOp(
        Path(components_path / component_names[2]), component_args
    )

    pipeline.add_op(third_component_op, dependencies=second_component_op)
    pipeline.add_op(first_component_op)
    pipeline.add_op(second_component_op, dependencies=first_component_op)

    pipeline.sort_graph()
    assert list(pipeline._graph.keys()) == [
        "First component",
        "Second component",
        "Third component",
    ]
    assert pipeline._graph["First component"]["dependencies"] == []
    assert pipeline._graph["Second component"]["dependencies"] == ["First component"]
    assert pipeline._graph["Third component"]["dependencies"] == ["Second component"]

    pipeline.compile()


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            ["first_component.yaml", "second_component.yaml", "third_component.yaml"],
        ),
    ],
)
def test_invalid_pipeline_dependencies(
    default_pipeline_args, valid_pipeline_example, tmp_path
):
    """
    Test that an InvalidPipelineDefinition exception is raised when attempting to create a pipeline
    with more than one operation defined without dependencies.
    """
    example_dir, component_names = valid_pipeline_example
    components_path = Path(valid_pipeline_path / example_dir)
    component_args = {"storage_args": "a dummy string arg"}

    pipeline = FondantPipeline(**default_pipeline_args)

    first_component_op = FondantComponentOp(
        Path(components_path / component_names[0]), component_args
    )
    second_component_op = FondantComponentOp(
        Path(components_path / component_names[1]), component_args
    )
    third_component_op = FondantComponentOp(
        Path(components_path / component_names[2]), component_args
    )

    pipeline.add_op(third_component_op, dependencies=second_component_op)
    pipeline.add_op(second_component_op)
    with pytest.raises(InvalidPipelineDefinition):
        pipeline.add_op(first_component_op)


@pytest.mark.parametrize(
    "invalid_pipeline_example",
    [
        ("example_1", ["first_component.yaml", "second_component.yaml"]),
        ("example_2", ["first_component.yaml", "second_component.yaml"]),
    ],
)
def test_invalid_pipeline_compilation(
    default_pipeline_args, invalid_pipeline_example, tmp_path
):
    """
    Test that an InvalidPipelineDefinition exception is raised when attempting to compile
    an invalid pipeline definition.
    """
    example_dir, component_names = invalid_pipeline_example
    components_path = Path(invalid_pipeline_path / example_dir)
    component_args = {"storage_args": "a dummy string arg"}

    pipeline = FondantPipeline(**default_pipeline_args)

    first_component_op = FondantComponentOp(
        Path(components_path / component_names[0]), component_args
    )
    second_component_op = FondantComponentOp(
        Path(components_path / component_names[1]), component_args
    )

    pipeline.add_op(first_component_op)
    pipeline.add_op(second_component_op, dependencies=first_component_op)

    with pytest.raises(InvalidPipelineDefinition):
        pipeline.compile()


@pytest.mark.parametrize(
    "invalid_component_args",
    [
        {"invalid_arg": "a dummy string arg", "storage_args": "a dummy string arg"},
        {"args": 1, "storage_args": "a dummy string arg"},
    ],
)
def test_invalid_argument(default_pipeline_args, invalid_component_args, tmp_path):
    """
    Test that an exception is raised when the passed invalid argument name or type to the fondant
    component does not match the ones specified in the fondant specifications.
    """
    components_spec_path = Path(
        valid_pipeline_path / "example_1" / "first_component.yaml"
    )
    component_operation = FondantComponentOp(
        components_spec_path, invalid_component_args
    )

    pipeline = FondantPipeline(**default_pipeline_args)

    pipeline.add_op(component_operation)

    with pytest.raises((ValueError, TypeError)):
        pipeline.compile()
