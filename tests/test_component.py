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
    Component,
    DaskTransformComponent,
    LoadComponent,
    PandasTransformComponent,
    WriteComponent,
)
from fondant.data_io import DaskDataLoader, DaskDataWriter
from fondant.manifest import Manifest

components_path = Path(__file__).parent / "example_specs/components"
N_PARTITIONS = 2


def yaml_file_to_json_string(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        json_string = json.dumps(data)
    return json_string


@pytest.fixture
def patched_data_loading(monkeypatch):
    """Mock data loading so no actual data is loaded."""

    def mocked_load_dataframe(self):
        return dd.from_dict({"a": [1, 2, 3]}, npartitions=N_PARTITIONS)

    monkeypatch.setattr(DaskDataLoader, "load_dataframe", mocked_load_dataframe)


@pytest.fixture
def patched_data_writing(monkeypatch):
    """Mock data loading so no actual data is written."""

    def mocked_write_dataframe(self, dataframe):
        dataframe.compute()

    monkeypatch.setattr(DaskDataWriter, "write_dataframe", mocked_write_dataframe)
    monkeypatch.setattr(
        Component, "upload_manifest", lambda self, manifest, save_path: None
    )


def test_component_arguments():
    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(components_path / "arguments/input_manifest.json"),
        "--metadata",
        "{}",
        "--override_default_string_arg",
        "bar",
        "--output_manifest_path",
        str(components_path / "arguments/output_manifest.json"),
        "--component_spec",
        yaml_file_to_json_string(components_path / "arguments/component.yaml"),
    ]

    class MyComponent(Component):
        """Base component with dummy methods so it can be instantiated."""

        def _load_or_create_manifest(self) -> Manifest:
            pass

        def _process_dataset(self, manifest: Manifest) -> t.Union[None, dd.DataFrame]:
            pass

    component = MyComponent.from_args()
    assert component.user_arguments == {
        "string_default_arg": "foo",
        "integer_default_arg": 1,
        "float_default_arg": 3.14,
        "bool_default_arg": False,
        "list_default_arg": ["foo", "bar"],
        "dict_default_arg": {"foo": 1, "bar": 2},
        "override_default_string_arg": "bar",
    }


def test_load_component(patched_data_writing):
    # Mock CLI argumentsload
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

    class MyLoadComponent(LoadComponent):
        def load(self, *, flag, value):
            assert flag == "success"
            assert value == 1

            data = {
                "id": [0, 1],
                "captions_data": ["hello world", "this is another caption"],
            }
            return dd.DataFrame.from_dict(data, npartitions=N_PARTITIONS)

    component = MyLoadComponent.from_args()
    with mock.patch.object(MyLoadComponent, "load", wraps=component.load) as load:
        component.run()
        load.assert_called_once()


def test_dask_transform_component(patched_data_loading, patched_data_writing):
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

    class MyDaskComponent(DaskTransformComponent):
        def transform(self, dataframe, *, flag, value):
            assert flag == "success"
            assert value == 1
            assert isinstance(dataframe, dd.DataFrame)
            return dataframe

    component = MyDaskComponent.from_args()
    with mock.patch.object(
        MyDaskComponent, "transform", wraps=component.transform
    ) as transform:
        component.run()
        transform.assert_called_once()


def test_pandas_transform_component(patched_data_loading, patched_data_writing):
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
        def setup(self, *, flag, value):
            assert flag == "success"
            assert value == 1

        def transform(self, dataframe):
            assert isinstance(dataframe, pd.DataFrame)

    component = MyPandasComponent.from_args()
    setup = mock.patch.object(MyPandasComponent, "setup", wraps=component.setup)
    transform = mock.patch.object(
        MyPandasComponent, "transform", wraps=component.transform
    )
    with setup as setup, transform as transform:
        component.run()
        setup.assert_called_once()
        assert transform.call_count == N_PARTITIONS


def test_write_component(patched_data_loading):
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

    class MyWriteComponent(WriteComponent):
        def write(self, dataframe, *, flag, value):
            assert flag == "success"
            assert value == 1
            assert isinstance(dataframe, dd.DataFrame)

    component = MyWriteComponent.from_args()
    with mock.patch.object(MyWriteComponent, "write", wraps=component.write) as write:
        component.run()
        write.assert_called_once()
