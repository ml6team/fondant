"""Fondant pipelines test."""
import copy
from pathlib import Path

import pytest
import yaml
from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidPipelineDefinition
from fondant.pipeline import ComponentOp, Pipeline, Resources

valid_pipeline_path = Path(__file__).parent / "examples/pipelines/valid_pipeline"
invalid_pipeline_path = Path(__file__).parent / "examples/pipelines/invalid_pipeline"


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
            ["first_component", "second_component", "third_component"],
        ),
    ],
)
def test_component_op(
    valid_pipeline_example,
):
    component_args = {"storage_args": "a dummy string arg"}
    example_dir, component_names = valid_pipeline_example
    components_path = Path(valid_pipeline_path / example_dir)

    ComponentOp(
        Path(components_path / component_names[0]),
        arguments=component_args,
    )

    with pytest.raises(InvalidPipelineDefinition):
        ComponentOp(
            Path(components_path / component_names[0]),
            arguments=component_args,
            resources=Resources(
                node_pool_label="dummy_label",
            ),
        )

    with pytest.raises(InvalidPipelineDefinition):
        ComponentOp(
            Path(components_path / component_names[0]),
            arguments=component_args,
            resources=Resources(
                accelerator_number=1,
            ),
        )


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            ["first_component", "second_component", "third_component"],
        ),
    ],
)
def test_component_op_hash(
    valid_pipeline_example,
    monkeypatch,
):
    example_dir, component_names = valid_pipeline_example
    components_path = Path(valid_pipeline_path / example_dir)

    comp_0_op_spec_0 = ComponentOp(
        Path(components_path / component_names[0]),
        arguments={"storage_args": "a dummy string arg"},
    )

    comp_0_op_spec_1 = ComponentOp(
        Path(components_path / component_names[0]),
        arguments={"storage_args": "a different string arg"},
    )

    comp_1_op_spec_0 = ComponentOp(
        Path(components_path / component_names[1]),
        arguments={"storage_args": "a dummy string arg"},
    )

    comp_0_op_spec_0_copy = copy.deepcopy(comp_0_op_spec_0)

    assert (
        comp_0_op_spec_0.get_component_cache_key()
        != comp_0_op_spec_1.get_component_cache_key()
    )
    assert (
        comp_0_op_spec_0.get_component_cache_key()
        == comp_0_op_spec_0_copy.get_component_cache_key()
    )
    assert (
        comp_0_op_spec_0.get_component_cache_key()
        != comp_1_op_spec_0.get_component_cache_key()
    )


def test_component_op_caching_strategy(monkeypatch):
    components_path = Path(valid_pipeline_path / "example_1" / "first_component")
    for tag in ["latest", "dev", "1234"]:
        monkeypatch.setattr(
            ComponentSpec,
            "image",
            f"fndnt/test_component:{tag}",
        )
        comp_0_op_spec_0 = ComponentOp(
            components_path,
            arguments={"storage_args": "a dummy string arg"},
            cache=True,
        )
        if tag == "latest":
            assert comp_0_op_spec_0.cache is False
        else:
            assert comp_0_op_spec_0.cache is True


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            ["first_component", "second_component", "third_component"],
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
        "first_component",
        "second_component",
        "third_component",
    ]
    assert pipeline._graph["first_component"]["dependencies"] == []
    assert pipeline._graph["second_component"]["dependencies"] == ["first_component"]
    assert pipeline._graph["third_component"]["dependencies"] == ["second_component"]

    pipeline._validate_pipeline_definition("test_pipeline")


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            ["first_component", "second_component", "third_component"],
        ),
    ],
)
def test_invalid_pipeline_dependencies(default_pipeline_args, valid_pipeline_example):
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
        ("example_1", ["first_component", "second_component"]),
        ("example_2", ["first_component", "second_component"]),
        ("example_3", ["first_component", "second_component"]),
    ],
)
def test_invalid_pipeline_declaration(
    default_pipeline_args,
    invalid_pipeline_example,
):
    """Test that an InvalidPipelineDefinition exception is raised when attempting
    to register invalid components combinations.
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
        pipeline._validate_pipeline_definition("test_pipeline")


def test_invalid_pipeline_validation(default_pipeline_args):
    """
    Test that an InvalidPipelineDefinition exception is raised when attempting to compile
    an invalid pipeline definition.
    """
    components_path = Path(invalid_pipeline_path / "example_1")
    component_args = {"storage_args": "a dummy string arg"}

    first_component_op = ComponentOp(
        Path(components_path / "first_component"),
        arguments=component_args,
    )
    second_component_op = ComponentOp(
        Path(components_path / "second_component"),
        arguments=component_args,
    )

    # double dependency
    pipeline1 = Pipeline(**default_pipeline_args)
    pipeline1.add_op(first_component_op)
    with pytest.raises(InvalidPipelineDefinition):
        pipeline1.add_op(
            second_component_op,
            dependencies=[first_component_op, first_component_op],
        )

    # 2 components with no dependencies
    pipeline2 = Pipeline(**default_pipeline_args)
    pipeline2.add_op(first_component_op)
    with pytest.raises(InvalidPipelineDefinition):
        pipeline2.add_op(second_component_op)


def test_reusable_component_op():
    laion_retrieval_op = ComponentOp.from_registry(
        name="retrieve_laion_by_prompt",
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
    load_from_hub_default_op = ComponentOp.from_registry(
        name="load_from_hf_hub",
        arguments={
            "dataset_name": "test_dataset",
            "column_name_mapping": {"foo": "bar"},
            "image_column_names": None,
        },
    )

    load_from_hub_custom_op = ComponentOp(
        component_dir=load_from_hub_default_op.component_dir,
        arguments={
            "dataset_name": "test_dataset",
            "column_name_mapping": {"foo": "bar"},
            "image_column_names": None,
        },
    )

    assert (
        load_from_hub_custom_op.component_spec
        == load_from_hub_default_op.component_spec
    )


def test_pipeline_name():
    Pipeline(pipeline_name="valid-name", base_path="base_path")
    with pytest.raises(InvalidPipelineDefinition, match="The pipeline name violates"):
        Pipeline(pipeline_name="invalid name", base_path="base_path")
