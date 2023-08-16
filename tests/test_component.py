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
from fondant.executor import (
    DaskLoadExecutor,
    DaskTransformExecutor,
    DaskWriteExecutor,
    Executor,
    PandasTransformExecutor,
)
from fondant.manifest import Manifest

components_path = Path(__file__).parent / "example_specs/components"
N_PARTITIONS = 2


def yaml_file_to_json_string(file_path):
    with open(file_path) as file:
        data = yaml.safe_load(file)
        return json.dumps(data)


@pytest.fixture()
def _patched_data_loading(monkeypatch):
    """Mock data loading so no actual data is loaded."""

    def mocked_load_dataframe(self):
        return dd.from_dict({"images_data": [1, 2, 3]}, npartitions=N_PARTITIONS)

    monkeypatch.setattr(DaskDataLoader, "load_dataframe", mocked_load_dataframe)


@pytest.fixture()
def _patched_data_writing(monkeypatch):
    """Mock data loading so no actual data is written."""

    def mocked_write_dataframe(self, dataframe):
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


def test_component_arguments():
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "arguments/input_manifest.json"),
        "--metadata",
        "{}",
        "--output_manifest_path",
        str(components_path / "arguments/output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "arguments/component.yaml"),
        "--spec_mapping",
        '{"col_1": "col_2"}',
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
    expected_spec_mapping = {"col_1": "col_2"}
    assert executor.input_partition_rows == expected_partition_row_arg
    assert executor.spec_mapping == expected_spec_mapping
    assert executor.user_arguments == {
        "string_default_arg": "foo",
        "integer_default_arg": 0,
        "float_default_arg": 3.14,
        "bool_false_default_arg": False,
        "bool_true_default_arg": True,
        "list_default_arg": ["foo", "bar"],
        "dict_default_arg": {"foo": 1, "bar": 2},
        "string_default_arg_none": None,
        "integer_default_arg_none": None,
        "float_default_arg_none": None,
        "bool_default_arg_none": None,
        "list_default_arg_none": None,
        "dict_default_arg_none": None,
        "override_default_arg": "bar",
        "override_default_none_arg": 3.14,
        "override_default_arg_with_none": None,
    }


@pytest.mark.usefixtures("_patched_data_writing")
def test_load_component():
    # Mock CLI arguments load
    sys.argv = [
        "",
        "--metadata",
        json.dumps({"base_path": "/bucket", "run_id": "12345"}),
        "--flag",
        "success",
        "--value",
        "1",
        "--output_manifest_path",
        str(components_path / "output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
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

    executor = DaskLoadExecutor.from_args()
    assert executor.input_partition_rows is None
    load = patch_method_class(MyLoadComponent.load)
    with mock.patch.object(MyLoadComponent, "load", load):
        executor.execute(MyLoadComponent)
        load.mock.assert_called_once()


@pytest.mark.usefixtures("_patched_data_loading", "_patched_data_writing")
def test_dask_transform_component():
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "input_manifest.json"),
        "--metadata",
        "{}",
        "--flag",
        "success",
        "--value",
        "1",
        "--input_partition_rows",
        "disable",
        "--output_manifest_path",
        str(components_path / "output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
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

    executor = DaskTransformExecutor.from_args()
    assert executor.input_partition_rows == "disable"
    transform = patch_method_class(MyDaskComponent.transform)
    with mock.patch.object(
        MyDaskComponent,
        "transform",
        transform,
    ):
        executor.execute(MyDaskComponent)
        transform.mock.assert_called_once()


@pytest.mark.usefixtures("_patched_data_loading", "_patched_data_writing")
def test_pandas_transform_component():
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "input_manifest.json"),
        "--metadata",
        "{}",
        "--flag",
        "success",
        "--value",
        "1",
        "--output_manifest_path",
        str(components_path / "output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
    ]

    class MyPandasComponent(PandasTransformComponent):
        def __init__(self, *args, flag, value):
            assert flag == "success"
            assert value == 1

        def transform(self, dataframe):
            assert isinstance(dataframe, pd.DataFrame)
            return dataframe.rename(columns={"images": "embeddings"})

    executor = PandasTransformExecutor.from_args()
    init = patch_method_class(MyPandasComponent.__init__)
    transform = patch_method_class(MyPandasComponent.transform)
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
def test_write_component():
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "input_manifest.json"),
        "--metadata",
        "{}",
        "--flag",
        "success",
        "--value",
        "1",
        "--component_spec",
        yaml_file_to_json_string(components_path / "component.yaml"),
    ]

    class MyWriteComponent(DaskWriteComponent):
        def __init__(self, *args, flag, value):
            self.flag = flag
            self.value = value

        def write(self, dataframe):
            assert self.flag == "success"
            assert self.value == 1
            assert isinstance(dataframe, dd.DataFrame)

    executor = DaskWriteExecutor.from_args()
    write = patch_method_class(MyWriteComponent.write)
    with mock.patch.object(MyWriteComponent, "write", write):
        executor.execute(MyWriteComponent)
        write.mock.assert_called_once()
