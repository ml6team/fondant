import json
import sys
import typing as t
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import pandas as pd
import pytest
import yaml
from fondant.component import (
    DaskLoadComponent,
    DaskTransformComponent,
    DaskWriteComponent,
    PandasTransformComponent,
)
from fondant.component_spec import ComponentSpec
from fondant.data_io import DaskDataLoader, DaskDataWriter
from fondant.executor import Executor, ExecutorFactory, PandasTransformExecutor
from fondant.manifest import Manifest, Metadata

components_path = Path(__file__).parent / "example_specs/components"
base_path = Path(__file__).parent / "example_specs/mock_base_path"

N_PARTITIONS = 2


def yaml_file_to_json_string(file_path):
    with open(file_path) as file:
        data = yaml.safe_load(file)
        return json.dumps(data)


@pytest.fixture()
def metadata():
    return Metadata(
        pipeline_name="example_pipeline",
        base_path=str(base_path),
        component_id="component_2",
        run_id="example_pipeline_2024",
        cache_key="42",
    )


@pytest.fixture()
def _patched_data_loading(monkeypatch):
    """Mock data loading so no actual data is loaded."""

    def mocked_load_dataframe(self):
        return dd.from_dict({"images_data": [1, 2, 3]}, npartitions=N_PARTITIONS)

    monkeypatch.setattr(DaskDataLoader, "load_dataframe", mocked_load_dataframe)


@pytest.fixture()
def _patched_data_writing(monkeypatch):
    """Mock data loading so no actual data is written."""

    def mocked_write_dataframe(self, dataframe, dask_client=None):
        dataframe.compute()

    monkeypatch.setattr(DaskDataWriter, "write_dataframe", mocked_write_dataframe)
    monkeypatch.setattr(
        Executor,
        "upload_manifest",
        lambda self, manifest, save_path: None,
    )


def patch_method_class(method):
    """Patch a method on a class instead of an instance. The returned method can be passed to
    `mock.patch.object` as the `wraps` argument.
    """
    m = mock.MagicMock()

    def wrapper(self, *args, **kwargs):
        m(*args, **kwargs)
        return method(self, *args, **kwargs)

    wrapper.mock = m
    return wrapper


def test_component_arguments(metadata):
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "arguments/input_manifest.json"),
        "--metadata",
        metadata.to_json(),
        "--output_manifest_path",
        str(components_path / "arguments/output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "arguments/component.yaml"),
        "--cache",
        "True",
        "--input_partition_rows",
        "100",
        "--override_default_arg",
        "bar",
        "--override_default_none_arg",
        "3.14",
        "--override_default_arg_with_none",
        "None",
    ]

    class MyExecutor(Executor):
        """Base component with dummy methods so it can be instantiated."""

        def _load_or_create_manifest(self) -> Manifest:
            pass

        def _process_dataset(self, manifest: Manifest) -> t.Union[None, dd.DataFrame]:
            pass

    executor = MyExecutor.from_args()
    expected_partition_row_arg = 100
    assert executor.input_partition_rows == expected_partition_row_arg
    assert executor.cache is True
    assert executor.user_arguments == {
        "integer_default_arg": 0,
        "float_default_arg": 3.14,
        "bool_false_default_arg": False,
        "bool_true_default_arg": True,
        "list_default_arg": ["foo", "bar"],
        "dict_default_arg": {"foo": 1, "bar": 2},
        "string_default_arg": "foo",
        "string_default_arg_none": None,
        "integer_default_arg_none": 0,
        "float_default_arg_none": 0.0,
        "bool_default_arg_none": False,
        "list_default_arg_none": [],
        "dict_default_arg_none": {},
        "override_default_arg": "bar",
        "override_default_arg_with_none": None,
        "optional_arg": None,
    }


def test_run_with_cache(metadata, monkeypatch):
    input_manifest_path = str(components_path / "arguments/input_manifest.json")
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        input_manifest_path,
        "--metadata",
        metadata.to_json(),
        "--output_manifest_path",
        str(components_path / "arguments/output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "arguments/component.yaml"),
        "--cache",
        "True",
        "--input_partition_rows",
        "100",
        "--override_default_arg",
        "bar",
        "--override_default_none_arg",
        "3.14",
        "--override_default_arg_with_none",
        "None",
        "--cluster_type" "local" "--client_kwargs" "{}",
    ]

    class MyExecutor(Executor):
        """Base component with dummy methods so it can be instantiated."""

        def _load_or_create_manifest(self) -> Manifest:
            pass

        def _process_dataset(self, manifest: Manifest) -> t.Union[None, dd.DataFrame]:
            pass

    executor = MyExecutor.from_args()
    matching_execution_manifest = executor._get_cached_manifest()
    # Check that the latest manifest is returned
    assert matching_execution_manifest.run_id == "example_pipeline_2023"
    # Check that the previous component is cached due to similar run IDs
    assert executor._is_previous_cached(Manifest.from_file(input_manifest_path)) is True


def test_run_with_no_cache(metadata):
    input_manifest_path = str(components_path / "arguments/input_manifest.json")

    # Change metadata to a new cache key that's not cached
    metadata.cache_key = "123"
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        input_manifest_path,
        "--metadata",
        metadata.to_json(),
        "--output_manifest_path",
        str(components_path / "arguments/output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "arguments/component.yaml"),
        "--cache",
        "True",
        "--input_partition_rows",
        "100",
        "--override_default_arg",
        "bar",
        "--override_default_none_arg",
        "3.14",
        "--override_default_arg_with_none",
        "None",
    ]

    class MyExecutor(Executor):
        """Base component with dummy methods so it can be instantiated."""

        def _load_or_create_manifest(self) -> Manifest:
            pass

        def _process_dataset(self, manifest: Manifest) -> t.Union[None, dd.DataFrame]:
            pass

    executor = MyExecutor.from_args()
    matching_execution_manifest = executor._get_cached_manifest()
    # Check that the latest manifest is returned
    assert matching_execution_manifest is None


@pytest.mark.usefixtures("_patched_data_writing")
def test_load_component(metadata):
    # Mock CLI arguments load

    sys.argv = [
        "",
        "--metadata",
        metadata.to_json(),
        "--flag",
        "success",
        "--value",
        "1",
        "--output_manifest_path",
        str(components_path / "output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
        "--cache",
        "False",
    ]

    class MyLoadComponent(DaskLoadComponent):
        def __init__(self, *args, flag, value):
            self.flag = flag
            self.value = value

        def load(self):
            assert self.flag == "success"
            assert self.value == 1
            data = {
                "id": [0, 1],
                "captions_data": ["hello world", "this is another caption"],
            }
            return dd.DataFrame.from_dict(data, npartitions=N_PARTITIONS)

    executor_factory = ExecutorFactory(MyLoadComponent)
    executor = executor_factory.get_executor()
    assert executor.input_partition_rows is None

    load = patch_method_class(MyLoadComponent.load)
    with mock.patch.object(MyLoadComponent, "load", load):
        executor.execute(MyLoadComponent)
        load.mock.assert_called_once()


@pytest.mark.usefixtures("_patched_data_loading", "_patched_data_writing")
def test_dask_transform_component(metadata):
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "input_manifest.json"),
        "--metadata",
        metadata.to_json(),
        "--flag",
        "success",
        "--value",
        "1",
        "--input_partition_rows",
        "10",
        "--output_manifest_path",
        str(components_path / "output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
        "--cache",
        "False",
    ]

    class MyDaskComponent(DaskTransformComponent):
        def __init__(self, *args, flag, value):
            self.flag = flag
            self.value = value

        def transform(self, dataframe):
            assert self.flag == "success"
            assert self.value == 1
            assert isinstance(dataframe, dd.DataFrame)
            return dataframe

    executor_factory = ExecutorFactory(MyDaskComponent)
    executor = executor_factory.get_executor()
    expected_input_partition_rows = 10
    assert executor.input_partition_rows == expected_input_partition_rows
    transform = patch_method_class(MyDaskComponent.transform)
    with mock.patch.object(
        MyDaskComponent,
        "transform",
        transform,
    ):
        executor.execute(MyDaskComponent)
        transform.mock.assert_called_once()


@pytest.mark.usefixtures("_patched_data_loading", "_patched_data_writing")
def test_pandas_transform_component(metadata):
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "input_manifest.json"),
        "--metadata",
        metadata.to_json(),
        "--flag",
        "success",
        "--value",
        "1",
        "--output_manifest_path",
        str(components_path / "output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
        "--cache",
        "False",
    ]

    class MyPandasComponent(PandasTransformComponent):
        def __init__(self, *args, flag, value):
            assert flag == "success"
            assert value == 1

        def transform(self, dataframe):
            assert isinstance(dataframe, pd.DataFrame)
            return dataframe.rename(columns={"images": "embeddings"})

    init = patch_method_class(MyPandasComponent.__init__)
    transform = patch_method_class(MyPandasComponent.transform)
    executor_factory = ExecutorFactory(MyPandasComponent)
    executor = executor_factory.get_executor()
    with mock.patch.object(MyPandasComponent, "__init__", init), mock.patch.object(
        MyPandasComponent,
        "transform",
        transform,
    ):
        executor.execute(MyPandasComponent)
        init.mock.assert_called_once()
        assert transform.mock.call_count == N_PARTITIONS


def test_wrap_transform():
    """
    Test wrapped transform for:
    - Converting between hierarchical and flat columns
    - Trimming columns not specified in `produces`
    - Ordering columns according to specification (so `map_partitions` does not fail).
    """
    spec = ComponentSpec(
        {
            "name": "Test component",
            "description": "Component for testing",
            "image": "component:test",
            "consumes": {
                "image": {
                    "fields": {
                        "height": {
                            "type": "int16",
                        },
                        "width": {
                            "type": "int16",
                        },
                    },
                },
                "caption": {
                    "fields": {
                        "text": {
                            "type": "string",
                        },
                    },
                },
            },
            "produces": {
                "caption": {
                    "fields": {
                        "text": {
                            "type": "string",
                        },
                    },
                },
                "image": {
                    "fields": {
                        "height": {
                            "type": "int16",
                        },
                    },
                },
            },
        },
    )

    input_df = pd.DataFrame.from_dict(
        {
            "image_height": [0, 1, 2],
            "image_width": [3, 4, 5],
            "caption_text": ["one", "two", "three"],
        },
    )

    def transform(dataframe: pd.DataFrame) -> pd.DataFrame:
        # Check hierarchical columns
        assert dataframe.columns.tolist() == [
            ("image", "height"),
            ("image", "width"),
            ("caption", "text"),
        ]
        return dataframe

    wrapped_transform = PandasTransformExecutor.wrap_transform(transform, spec=spec)
    output_df = wrapped_transform(input_df)

    # Check column flattening, trimming, and ordering
    assert output_df.columns.tolist() == ["caption_text", "image_height"]


@pytest.mark.usefixtures("_patched_data_loading")
def test_write_component(metadata):
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "input_manifest.json"),
        "--metadata",
        metadata.to_json(),
        "--flag",
        "success",
        "--value",
        "1",
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
        "--cache",
        "False",
    ]

    class MyWriteComponent(DaskWriteComponent):
        def __init__(self, *args, flag, value):
            self.flag = flag
            self.value = value

        def write(self, dataframe):
            assert self.flag == "success"
            assert self.value == 1
            assert isinstance(dataframe, dd.DataFrame)

    executor_factory = ExecutorFactory(MyWriteComponent)
    executor = executor_factory.get_executor()
    write = patch_method_class(MyWriteComponent.write)
    with mock.patch.object(MyWriteComponent, "write", write):
        executor.execute(MyWriteComponent)
        write.mock.assert_called_once()
