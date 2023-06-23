"""Fondant pipelines test."""
from pathlib import Path

import pytest
import yaml

from fondant.exceptions import InvalidPipelineDefinition
from fondant.pipeline import ComponentOp, ComponentSpec, Pipeline

valid_pipeline_path = Path(__file__).parent / "example_pipelines/valid_pipeline"
invalid_pipeline_path = Path(__file__).parent / "example_pipelines/invalid_pipeline"
custom_spec_path = (
    Path(__file__).parent / "example_pipelines/load_from_hub_custom_spec.yaml"
)


def yaml_file_to_dict(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)


@pytest.fixture()
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
def test_valid_pipeline(
    default_pipeline_args,
    valid_pipeline_example,
    tmp_path,
    monkeypatch,
):
    """Test that a valid pipeline definition can be compiled without errors."""
    example_dir, component_names = valid_pipeline_example
    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(valid_pipeline_path / example_dir)

    pipeline = Pipeline(**default_pipeline_args)

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", str(tmp_path / "test_pipeline.tgz"))

    first_component_op = ComponentOp(
        Path(components_path / component_names[0]),
        arguments=component_args,
    )
    second_component_op = ComponentOp(
        Path(components_path / component_names[1]),
        arguments=component_args,
    )
    third_component_op = ComponentOp(
        Path(components_path / component_names[2]),
        arguments=component_args,
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
    default_pipeline_args,
    valid_pipeline_example,
    tmp_path,
):
    """
    Test that an InvalidPipelineDefinition exception is raised when attempting to create a pipeline
    with more than one operation defined without dependencies.
    """
    example_dir, component_names = valid_pipeline_example
    components_path = Path(valid_pipeline_path / example_dir)
    component_args = {"storage_args": "a dummy string arg"}

    pipeline = Pipeline(**default_pipeline_args)

    first_component_op = ComponentOp(
        Path(components_path / component_names[0]),
        arguments=component_args,
    )
    second_component_op = ComponentOp(
        Path(components_path / component_names[1]),
        arguments=component_args,
    )
    third_component_op = ComponentOp(
        Path(components_path / component_names[2]),
        arguments=component_args,
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
    default_pipeline_args,
    invalid_pipeline_example,
    tmp_path,
):
    """
    Test that an InvalidPipelineDefinition exception is raised when attempting to compile
    an invalid pipeline definition.
    """
    example_dir, component_names = invalid_pipeline_example
    components_path = Path(invalid_pipeline_path / example_dir)
    component_args = {"storage_args": "a dummy string arg"}

    pipeline = Pipeline(**default_pipeline_args)

    first_component_op = ComponentOp(
        Path(components_path / component_names[0]),
        arguments=component_args,
    )
    second_component_op = ComponentOp(
        Path(components_path / component_names[1]),
        arguments=component_args,
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
        valid_pipeline_path / "example_1" / "first_component.yaml",
    )
    component_operation = ComponentOp(
        components_spec_path,
        arguments=invalid_component_args,
    )

    pipeline = Pipeline(**default_pipeline_args)

    pipeline.add_op(component_operation)

    with pytest.raises((ValueError, TypeError)):
        pipeline.compile()


def test_reusable_component_op():
    laion_retrieval_op = ComponentOp.from_registry(
        name="prompt_based_laion_retrieval",
        arguments={"num_images": 2, "aesthetic_score": 9, "aesthetic_weight": 0.5},
    )
    assert laion_retrieval_op.component_spec, "component_spec_path could not be loaded"

    component_name = "this_component_does_not_exist"
    with pytest.raises(
        ValueError,
        match=f"No reusable component with name {component_name} " "found.",
    ):
        ComponentOp.from_registry(
            name=component_name,
        )


def test_defining_reusable_component_op_with_custom_spec():
    load_from_hub_op_default_op = ComponentOp.from_registry(
        name="load_from_hf_hub",
        arguments={
            "dataset_name": "test_dataset",
            "column_name_mapping": {"foo": "bar"},
            "image_column_names": None,
        },
    )

    load_from_hub_op_default_spec = ComponentSpec(
        yaml_file_to_dict(load_from_hub_op_default_op.component_spec_path),
    )

    load_from_hub_op_custom_op = ComponentOp.from_registry(
        name="load_from_hf_hub",
        component_spec_path=custom_spec_path,
        arguments={
            "dataset_name": "test_dataset",
            "column_name_mapping": {"foo": "bar"},
            "image_column_names": None,
        },
    )

    load_from_hub_op_custom_spec = ComponentSpec(yaml_file_to_dict(custom_spec_path))

    assert load_from_hub_op_custom_op.component_spec == load_from_hub_op_custom_spec
    assert load_from_hub_op_default_op.component_spec == load_from_hub_op_default_spec
    assert load_from_hub_op_default_op.component_spec != load_from_hub_op_custom_spec
    assert load_from_hub_op_custom_op.component_spec != load_from_hub_op_default_spec
