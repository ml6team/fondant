"""Fondant pipelines test."""
import copy
from pathlib import Path

import pyarrow as pa
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
        "name": "pipeline",
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
            [
                "first_component",
                "second_component",
                "third_component",
                "fourth_component",
            ],
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

    dataset = pipeline.read(
        Path(components_path / component_names[0]),
        arguments=component_args,
        produces={"images_array": pa.binary()},
    )
    dataset = dataset.apply(
        Path(components_path / component_names[1]),
        arguments=component_args,
        consumes={"images_data": "images_array"},
        produces={"embeddings_data": "embeddings_array"},
    )
    dataset = dataset.apply(
        Path(components_path / component_names[2]),
        arguments=component_args,
        consumes={"images_data": "images_array", "embeddings_data": "embeddings_array"},
    )
    dataset.apply(
        Path(components_path / component_names[3]),
        arguments=component_args,
        consumes={"images_array": pa.binary()},
    )

    pipeline.sort_graph()
    assert list(pipeline._graph.keys()) == [
        "first_component",
        "second_component",
        "third_component",
        "fourth_component",
    ]
    assert pipeline._graph["first_component"]["dependencies"] == []
    assert pipeline._graph["second_component"]["dependencies"] == ["first_component"]
    assert pipeline._graph["third_component"]["dependencies"] == ["second_component"]
    assert pipeline._graph["fourth_component"]["dependencies"] == ["third_component"]

    pipeline._validate_pipeline_definition("test_pipeline")


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            [
                "first_component",
                "second_component",
                "third_component",
                "fourth_component",
            ],
        ),
    ],
)
def test_invalid_pipeline_schema(
    default_pipeline_args,
    valid_pipeline_example,
    tmp_path,
    monkeypatch,
):
    """Test that valid pipeline errors are returned when defining invalid pipeline schemas."""
    example_dir, component_names = valid_pipeline_example
    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(valid_pipeline_path / example_dir)

    default_valid_spec = {
        0: {
            "produces": {"images_array": pa.binary()},
        },
        1: {
            "consumes": {"images_array": "images_data"},
            "produces": {"embeddings_data": "embeddings_array"},
        },
        2: {
            "consumes": {
                "images_array": "images_data",
                "embeddings_array": "embeddings_data",
            },
        },
        3: {
            "consumes": {"images_array": "images_array"},
        },
    }

    invalid_specs = [
        # "images_pictures" does not exist in the dataset
        {
            1: {
                "consumes": {"images_pictures": "images_data"},
                "produces": {"embeddings_data": "embeddings_array"},
            },
        },
        # "images_array" does not exist in the component spec
        {
            1: {
                "consumes": {"images_pictures": "images_array"},
                "produces": {"embeddings_data": "embeddings_array"},
            },
        },
        # Extra field in the produces mapping that does not have a corresponding field for a
        # non-generic component
        {
            1: {
                "consumes": {
                    "images_pictures": "images_array",
                    "non_existent_field": "non_existent_field",
                },
                "produces": {"embeddings_data": "embeddings_array"},
            },
        },
        # A custom field is defined in the produced mapping of a generic component which
        # already exists in the dataset
        {
            3: {
                "consumes": {"embeddings_data": "embeddings_field"},
            },
        },
    ]

    for invalid_spec in invalid_specs:
        spec = copy.deepcopy(default_valid_spec)
        spec.update(invalid_spec)
        pipeline = Pipeline(**default_pipeline_args)
        # override the default package_path with temporary path to avoid the creation of artifacts
        monkeypatch.setattr(
            pipeline,
            "package_path",
            str(tmp_path / "test_pipeline.tgz"),
        )

        dataset = pipeline.read(
            Path(components_path / component_names[0]),
            arguments=component_args,
            produces=spec[0]["produces"],
        )
        dataset = dataset.apply(
            Path(components_path / component_names[1]),
            arguments=component_args,
            consumes=spec[1]["consumes"],
            produces=spec[1]["produces"],
        )
        dataset = dataset.apply(
            Path(components_path / component_names[2]),
            arguments=component_args,
            consumes=spec[2]["consumes"],
        )
        dataset.apply(
            Path(components_path / component_names[3]),
            arguments=component_args,
            consumes=spec[3]["consumes"],
        )
        with pytest.raises(InvalidPipelineDefinition):
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

    dataset = pipeline.read(
        Path(components_path / component_names[0]),
        arguments=component_args,
        produces={"image_data": pa.binary()},
    )
    dataset = dataset.apply(
        Path(components_path / component_names[1]),
        arguments=component_args,
    )
    with pytest.raises(InvalidPipelineDefinition):
        pipeline.read(
            Path(components_path / component_names[2]),
            arguments=component_args,
            produces={"image_data": pa.binary()},
        )


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

    dataset = pipeline.read(
        Path(components_path / component_names[0]),
        arguments=component_args,
        produces={"image_data": pa.binary()},
    )
    dataset.apply(
        Path(components_path / component_names[1]),
        arguments=component_args,
    )

    with pytest.raises(InvalidPipelineDefinition):
        pipeline._validate_pipeline_definition("test_pipeline")


def test_reusable_component_op():
    laion_retrieval_op = ComponentOp(
        name_or_path="retrieve_laion_by_prompt",
        arguments={"num_images": 2, "aesthetic_score": 9, "aesthetic_weight": 0.5},
    )
    assert laion_retrieval_op.component_spec, "component_spec_path could not be loaded"

    component_name = "this_component_does_not_exist"
    with pytest.raises(
        ValueError,
        match=f"No reusable component with name {component_name} " "found.",
    ):
        ComponentOp(component_name)


def test_defining_reusable_component_op_with_custom_spec():
    load_from_hub_default_op = ComponentOp(
        name_or_path="load_from_hf_hub",
        arguments={
            "dataset_name": "test_dataset",
            "column_name_mapping": {"foo": "bar"},
            "image_column_names": None,
        },
    )

    load_from_hub_custom_op = ComponentOp(
        name_or_path=load_from_hub_default_op.component_dir,
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
    Pipeline(name="valid-name", base_path="base_path")
    with pytest.raises(InvalidPipelineDefinition, match="The pipeline name violates"):
        Pipeline(name="invalid name", base_path="base_path")
