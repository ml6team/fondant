"""Fondant pipelines test."""
import copy
import re
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from fondant.component import DaskLoadComponent
from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidPipelineDefinition
from fondant.core.schema import Field, Type
from fondant.dataset import (
    ComponentOp,
    Image,
    Pipeline,
    Resources,
    lightweight_component,
)

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

    ComponentOp.from_component_yaml(
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


def test_component_op_python_component(default_pipeline_args):
    @lightweight_component()
    class Foo(DaskLoadComponent):
        def load(self) -> dd.DataFrame:
            df = pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                },
                index=pd.Index(["a", "b", "c"], name="id"),
            )
            return dd.from_pandas(df, npartitions=1)

    component = ComponentOp.from_ref(Foo, produces={"bar": pa.string()})
    assert component.component_spec._specification == {
        "name": "Foo",
        "image": Image.resolve_fndnt_base_image(),
        "description": "lightweight component",
        "consumes": {"additionalProperties": True},
        "produces": {"additionalProperties": True},
    }


def test_component_op_bad_ref():
    with pytest.raises(
        ValueError,
        match="""Invalid reference type: <class 'int'>.
                Expected a string, Path, or a lightweight component class.""",
    ):
        ComponentOp.from_ref(123)


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

    comp_0_op_spec_0 = ComponentOp.from_component_yaml(
        Path(components_path / component_names[0]),
        arguments={"storage_args": "a dummy string arg"},
    )

    comp_0_op_spec_1 = ComponentOp.from_component_yaml(
        Path(components_path / component_names[0]),
        arguments={"storage_args": "a different string arg"},
    )

    comp_1_op_spec_0 = ComponentOp.from_component_yaml(
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
        comp_0_op_spec_0 = ComponentOp.from_component_yaml(
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


def test_invalid_pipeline_schema(
    default_pipeline_args,
    tmp_path,
    monkeypatch,
):
    """Test that valid pipeline errors are returned when defining invalid pipeline schemas."""
    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(valid_pipeline_path / "example_1")

    pipeline = Pipeline(**default_pipeline_args)
    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(
        pipeline,
        "package_path",
        str(tmp_path / "test_pipeline.tgz"),
    )

    dataset = pipeline.read(
        Path(components_path / "first_component"),
        arguments=component_args,
        produces={"images_array": pa.binary()},
    )

    # "images_pictures" does not exist in the dataset
    with pytest.raises(InvalidPipelineDefinition):
        dataset.apply(
            Path(components_path / "second_component"),
            arguments=component_args,
            consumes={"images_pictures": "images_array"},
        )

    # "images_array" does not exist in the component spec
    with pytest.raises(InvalidPipelineDefinition):
        dataset.apply(
            Path(components_path / "second_component"),
            arguments=component_args,
            consumes={"images_array": "images_array"},
        )

    # Extra field in the consumes mapping that does not have a corresponding field
    # in the dataset
    with pytest.raises(InvalidPipelineDefinition):
        dataset.apply(
            Path(components_path / "second_component"),
            arguments=component_args,
            consumes={
                "images_pictures": "images_array",
                "non_existent_field": "non_existent_field",
            },
            produces={"embeddings_data": "embeddings_array"},
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
    laion_retrieval_op = ComponentOp.from_component_yaml(
        "retrieve_laion_by_prompt",
        arguments={"num_images": 2, "aesthetic_score": 9, "aesthetic_weight": 0.5},
    )
    assert laion_retrieval_op.component_spec, "component_spec_path could not be loaded"

    component_name = "this_component_does_not_exist"
    with pytest.raises(
        ValueError,
        match=f"No reusable component with name {component_name} " "found.",
    ):
        ComponentOp.from_component_yaml(component_name)


def test_defining_reusable_component_op_with_custom_spec():
    load_from_hub_default_op = ComponentOp.from_component_yaml(
        "load_from_hf_hub",
        arguments={
            "dataset_name": "test_dataset",
            "column_name_mapping": {"foo": "bar"},
            "image_column_names": None,
        },
    )

    load_from_hub_custom_op = ComponentOp.from_component_yaml(
        load_from_hub_default_op.component_dir,
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


def test_schema_propagation():
    """Test that the schema is propagated correctly between datasets taking into account the
    component specs and `consumes` and `produces` arguments.
    """
    pipeline = Pipeline(name="pipeline", base_path="base_path")

    pipeline.get_run_id = lambda: "pipeline-id"

    dataset = pipeline.read(
        "load_from_hf_hub",
        produces={
            "image": pa.binary(),
        },
    )

    assert dataset.fields == {
        "image": Field(
            "image",
            type=Type(pa.binary()),
            location="/pipeline-id/load_from_hugging_face_hub",
        ),
    }

    dataset = dataset.apply(
        "caption_images",
    )

    assert dataset.fields == {
        "image": Field(
            "image",
            type=Type(pa.binary()),
            location="/pipeline-id/load_from_hugging_face_hub",
        ),
        "caption": Field(
            "caption",
            type=Type(pa.string()),
            location="/pipeline-id/caption_images",
        ),
    }

    dataset = dataset.apply(
        "filter_language",
        consumes={
            "text": "caption",
        },
    )

    assert dataset.fields == {
        "image": Field(
            "image",
            type=Type(pa.binary()),
            location="/pipeline-id/load_from_hugging_face_hub",
        ),
        "caption": Field(
            "caption",
            type=Type(pa.string()),
            location="/pipeline-id/caption_images",
        ),
    }

    dataset = dataset.apply(
        "chunk_text",
        consumes={
            "text": "caption",
        },
        produces={
            "text": "chunks",
        },
    )

    assert dataset.fields == {
        "chunks": Field(
            "chunks",
            type=Type(pa.string()),
            location="/pipeline-id/chunk_text",
        ),
        "original_document_id": Field(
            "original_document_id",
            type=Type(pa.string()),
            location="/pipeline-id/chunk_text",
        ),
    }


def test_invoked_field_schema_raise_exception():
    """Test that check if the invoked field schema not matches the
    current schema raise an InvalidPipelineDefinition.
    """
    pipeline = Pipeline(name="pipeline", base_path="base_path")

    pipeline.get_run_id = lambda: "pipeline-id"

    dataset = pipeline.read(
        "load_from_hf_hub",
        produces={
            "image": pa.binary(),
        },
    )

    dataset.write(
        "write_to_file",
        consumes={
            "image": pa.string(),
        },
    )

    expected_error_msg = re.escape(
        "The invoked field 'image' of the 'write_to_file' component does not match the previously "
        "created field type.\n The 'image' field is currently defined with the following "
        "type:\nType(DataType(binary))\nThe current component to trying to invoke "
        "it with this type:\nType(DataType(string))",
    )

    with pytest.raises(InvalidPipelineDefinition, match=expected_error_msg):
        pipeline.validate("pipeline-id")


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
def test_infer_consumes_if_not_defined(
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

    # Empty consumes & additionalProperties=False -> infer component spec defined columns
    assert list(dataset.fields.keys()) == ["images_array"]
    dataset = dataset.apply(
        Path(components_path / component_names[1]),
        arguments=component_args,
    )

    assert dataset.pipeline._graph["second_component"][
        "operation"
    ].operation_spec.to_dict()["consumes"] == {
        "images_data": {"type": "binary"},
    }

    # Empty consumes, additionalProperties=False, two consumes fields in component spec defined
    assert list(dataset.fields.keys()) == ["images_array", "embeddings_data"]
    dataset = dataset.apply(
        Path(components_path / component_names[2]),
        arguments=component_args,
    )

    assert dataset.pipeline._graph["third_component"][
        "operation"
    ].operation_spec.to_dict()["consumes"] == {
        "images_data": {"type": "binary"},
        "embeddings_data": {"items": {"type": "float32"}, "type": "array"},
    }

    # Additional properties is true, no consumes field in dataset apply
    # -> infer operation spec, load all columns of dataset (images_data, embeddings_data)
    assert list(dataset.fields.keys()) == [
        "images_array",
        "embeddings_data",
        "images_data",
    ]
    dataset = dataset.apply(
        Path(components_path / component_names[3]),
        arguments=component_args,
    )

    assert dataset.pipeline._graph["fourth_component"][
        "operation"
    ].operation_spec.to_dict()["consumes"] == {
        "images_data": {"type": "binary"},
        "images_array": {"type": "binary"},
        "embeddings_data": {"items": {"type": "float32"}, "type": "array"},
    }


def test_consumes_name_to_name_mapping(
    default_pipeline_args,
    tmp_path,
    monkeypatch,
):
    """Test that a valid pipeline definition can be compiled without errors."""
    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(valid_pipeline_path / "example_1")
    pipeline = Pipeline(**default_pipeline_args)

    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", str(tmp_path / "test_pipeline.tgz"))

    dataset = pipeline.read(
        Path(components_path / "first_component"),
        arguments=component_args,
        produces={"images_data": pa.binary(), "second_field": pa.string()},
    )

    dataset.apply(
        Path(components_path / "fourth_component"),
        arguments=component_args,
        consumes={"images_data": "images_data"},
    )

    assert dataset.pipeline._graph["fourth_component"][
        "operation"
    ].operation_spec.to_dict()["consumes"] == {
        "images_data": {"type": "binary"},
    }
