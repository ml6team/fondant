"""Fondant pipelines caching test."""
from pathlib import Path

import pytest
from fondant.pipeline import ComponentOp, Pipeline

valid_pipeline_path = Path(__file__).parent / "example_pipelines/valid_pipeline"
invalid_pipeline_path = Path(__file__).parent / "example_pipelines/invalid_pipeline"
base_path = Path(__file__).parent / "example_pipelines/mock_base_path"


@pytest.fixture()
def default_pipeline_args():
    return {
        "pipeline_name": "pipeline",
        "base_path": base_path,
    }


@pytest.fixture()
def mocked_component_op(monkeypatch):
    # Define the mock method
    def mocked_get_component_image_hash(self, image_ref):
        return "sha:42"

    # Apply the monkeypatch to replace the original method with the mocked method
    monkeypatch.setattr(
        ComponentOp,
        "get_component_image_hash",
        mocked_get_component_image_hash,
    )

    # Return a function that takes custom arguments and returns an instance of ComponentOp
    def create_mocked_component_op(component_dir, **kwargs):
        return ComponentOp(component_dir=component_dir, **kwargs)

    return create_mocked_component_op


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            ["first_component", "second_component", "third_component"],
            [
                "35a7ce177f3daefd402f18192f2bab51",
                "2bbcf92296fc86856c4b5bebb4562bf9",
                "be724a135ccfc72b30e69f4990beb993",
            ],
        ),
    ],
)
def test_cached_execution_default_args(
    default_pipeline_args,
    mocked_component_op,
    valid_pipeline_example,
    tmp_path,
    monkeypatch,
):
    """Test that the execution of components that have already been executed is cached."""
    example_dir, component_names, cache_keys_default = valid_pipeline_example
    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(valid_pipeline_path / example_dir)

    pipeline = Pipeline(**default_pipeline_args)
    pipeline_package_path = str(tmp_path / "test_pipeline.tgz")
    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", pipeline_package_path)

    first_component_op = mocked_component_op(
        Path(components_path / component_names[0]),
        arguments=component_args,
    )
    second_component_op = mocked_component_op(
        Path(components_path / component_names[1]),
        arguments=component_args,
    )
    third_component_op = mocked_component_op(
        Path(components_path / component_names[2]),
        arguments=component_args,
    )

    pipeline.add_op(third_component_op, dependencies=second_component_op)
    pipeline.add_op(first_component_op)
    pipeline.add_op(second_component_op, dependencies=first_component_op)

    pipeline.sort_graph()

    for cache_disabled in [True, False]:
        cache_dict = pipeline.get_pipeline_cache_dict(cache_disabled=cache_disabled)

        assert cache_dict["first_component"]["cache_key"] == cache_keys_default[0]
        assert cache_dict["second_component"]["cache_key"] == cache_keys_default[1]
        assert cache_dict["third_component"]["cache_key"] == cache_keys_default[2]

        if cache_disabled:
            assert cache_dict["first_component"]["execute_component"] is True
            assert cache_dict["second_component"]["execute_component"] is True
            assert cache_dict["third_component"]["execute_component"] is True

        else:
            assert cache_dict["first_component"]["execute_component"] is False
            assert cache_dict["second_component"]["execute_component"] is False
            assert cache_dict["third_component"]["execute_component"] is False


@pytest.mark.parametrize(
    "valid_pipeline_example",
    [
        (
            "example_1",
            ["first_component", "second_component", "third_component"],
            [
                "35a7ce177f3daefd402f18192f2bab51",
                "2bbcf92296fc86856c4b5bebb4562bf9",
                "be724a135ccfc72b30e69f4990beb993",
            ],
        ),
    ],
)
def test_cached_execution_non_executed_second_component(
    default_pipeline_args,
    mocked_component_op,
    valid_pipeline_example,
    tmp_path,
    monkeypatch,
):
    """Test that the a proper cache dict is return when a component definition is changed."""
    example_dir, component_names, cache_keys_default = valid_pipeline_example
    component_args = {"storage_args": "a dummy string arg"}
    components_path = Path(valid_pipeline_path / example_dir)

    pipeline = Pipeline(**default_pipeline_args)
    pipeline_package_path = str(tmp_path / "test_pipeline.tgz")
    # override the default package_path with temporary path to avoid the creation of artifacts
    monkeypatch.setattr(pipeline, "package_path", pipeline_package_path)

    first_component_op = mocked_component_op(
        Path(components_path / component_names[0]),
        arguments=component_args,
    )
    second_component_op = mocked_component_op(
        Path(components_path / component_names[1]),
        arguments={"storage_args": "a changed string arg"},
    )
    third_component_op = mocked_component_op(
        Path(components_path / component_names[2]),
        arguments=component_args,
    )

    pipeline.add_op(third_component_op, dependencies=second_component_op)
    pipeline.add_op(first_component_op)
    pipeline.add_op(second_component_op, dependencies=first_component_op)

    pipeline.sort_graph()

    for cache_disabled in [True, False]:
        cache_dict = pipeline.get_pipeline_cache_dict(cache_disabled=cache_disabled)

        assert cache_dict["first_component"]["cache_key"] == cache_keys_default[0]
        assert cache_dict["second_component"]["cache_key"] != cache_keys_default[1]
        assert cache_dict["third_component"]["cache_key"] == cache_keys_default[2]

        if cache_disabled:
            assert cache_dict["first_component"]["execute_component"] is True
            assert cache_dict["second_component"]["execute_component"] is True
            assert cache_dict["third_component"]["execute_component"] is True

        else:
            assert cache_dict["first_component"]["execute_component"] is False
            assert cache_dict["second_component"]["execute_component"] is True
            assert cache_dict["third_component"]["execute_component"] is True
