"""Fondant component tests."""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import pytest
import yaml

from fondant.component import LoadComponent, TransformComponent, WriteComponent
from fondant.data_io import DaskDataLoader

components_path = Path(__file__).parent / "example_specs/components"
component_specs_path = Path(__file__).parent / "example_specs/component_specs"


def yaml_file_to_json_string(file_path):
    with open(file_path, "r") as file:
        # Load YAML data from file
        data = yaml.safe_load(file)

        # Convert to JSON string
        json_string = json.dumps(data)

    return json_string


class LoadFromHubComponent(LoadComponent):
    def load(self, args):
        data = {
            "id": [0, 1],
            "source": ["cloud", "cloud"],
            "captions_data": ["hello world", "this is another caption"],
        }

        df = dd.DataFrame.from_dict(data, npartitions=1)

        return df


# we mock the argparse arguments in order to run the tests without having to pass arguments
metadata = json.dumps({"base_path": ".", "run_id": "200"})


@mock.patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(
        input_manifest_path=".",
        output_manifest_path="result.parquet",
        metadata=metadata,
    ),
)
def test_component(mock_args):
    component = LoadFromHubComponent.from_file(
        component_specs_path / "valid_component.yaml"
    )

    # test component args
    assert component.input_manifest_path == "."
    assert component.output_manifest_path == "result.parquet"
    assert component.metadata == json.loads(metadata)

    # test custom args
    assert list(component.spec.args) == ["storage_args"]

    # test manifest
    initial_manifest = component._load_or_create_manifest()
    assert initial_manifest.metadata == {
        "base_path": ".",
        "run_id": "200",
        "component_id": "example_component",
    }


def test_transform_component(monkeypatch):
    """Test that arguments are passed correctly to `Component.transform` method and that valid
    errors are returned when required arguments are missing.
    """

    class EarlyStopException(Exception):
        """Used to stop execution early instead of mocking all later functionality."""

    # Mock `Dataset.load_dataframe` so no actual data is loaded
    def mocked_load_dataframe(self):
        return dd.from_dict({"a": [1, 2, 3]}, npartitions=1)

    monkeypatch.setattr(DaskDataLoader, "load_dataframe", mocked_load_dataframe)

    # Define paths to specs to instantiate component
    arguments_dir = components_path / "arguments"
    component_spec = arguments_dir / "component.yaml"
    input_manifest = arguments_dir / "input_manifest.json"

    component_spec_string = yaml_file_to_json_string(component_spec)

    # Implemented Component class
    class MyComponent(TransformComponent):
        def transform(self, dataframe, *, flag, value):
            assert flag == "success"
            assert value == 1
            raise EarlyStopException()

    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(input_manifest),
        "--metadata",
        "",
        "--flag",
        "success",
        "--value",
        "1",
        "--output_manifest_path",
        "",
        "--component_spec",
        f"{component_spec_string}",
    ]

    # Instantiate and run component
    component = MyComponent.from_args()

    with pytest.raises(EarlyStopException):
        component.run()

    # Remove component specs from arguments
    component_spec_index = sys.argv.index("--component_spec")
    del sys.argv[component_spec_index : component_spec_index + 2]

    # Instantiate and run component
    with pytest.raises(ValueError):
        MyComponent.from_args()


def test_write_component(tmp_path_factory, monkeypatch):
    """Test that arguments are passed correctly to `Component.write` method and that valid
    errors are returned when required arguments are missing.
    """

    # Mock `Dataset.load_dataframe` so no actual data is loaded
    def mocked_load_dataframe(self):
        return dd.from_dict({"a": [1, 2, 3]}, npartitions=1)

    monkeypatch.setattr(DaskDataLoader, "load_dataframe", mocked_load_dataframe)

    # Define paths to specs to instantiate component
    arguments_dir = components_path / "arguments"
    component_spec = arguments_dir / "component.yaml"
    input_manifest = arguments_dir / "input_manifest.json"

    component_spec_string = yaml_file_to_json_string(component_spec)

    # Implemented Component class
    class MyComponent(WriteComponent):
        def write(self, dataframe, *, flag, value):
            assert flag == "success"
            assert value == 1
            # Mock write function that sinks final data to a local directory
            with tmp_path_factory.mktemp("temp") as fn:
                dataframe.to_parquet(fn)

    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(input_manifest),
        "--metadata",
        "",
        "--flag",
        "success",
        "--value",
        "1",
        "--output_manifest_path",
        "",
        "--component_spec",
        f"{component_spec_string}",
    ]

    # # Instantiate and run component
    component = MyComponent.from_args()
    component.run()

    # Remove component specs from arguments
    component_spec_index = sys.argv.index("--component_spec")
    del sys.argv[component_spec_index : component_spec_index + 2]

    # Instantiate and run component
    with pytest.raises(ValueError):
        MyComponent.from_args()


def test_default_args_component(tmp_path_factory, monkeypatch):
    """Test that default arguments defined in the fondant spec are passed correctly and have the
    proper data type.
    """

    # Mock `Dataset.load_dataframe` so no actual data is loaded
    def mocked_load_dataframe(self):
        return dd.from_dict({"a": [1, 2, 3]}, npartitions=1)

    monkeypatch.setattr(DaskDataLoader, "load_dataframe", mocked_load_dataframe)

    # Define paths to specs to instantiate component
    arguments_dir = components_path / "arguments"
    component_spec = arguments_dir / "component_default_args.yaml"
    input_manifest = arguments_dir / "input_manifest.json"

    component_spec_string = yaml_file_to_json_string(component_spec)

    # Implemented Component class
    class MyComponent(WriteComponent):
        def write(
            self,
            dataframe,
            *,
            string_default_arg,
            integer_default_arg,
            float_default_arg,
            bool_default_arg,
            list_default_arg,
            dict_default_arg,
        ):
            float_const = 3.14
            # Mock write function that sinks final data to a local directory
            assert string_default_arg == "foo"
            assert integer_default_arg == 1
            assert float_default_arg == float_const
            assert bool_default_arg is False
            assert list_default_arg == ["foo", "bar"]
            assert dict_default_arg == {"foo": 1, "bar": 2}

    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(input_manifest),
        "--metadata",
        "",
        "--output_manifest_path",
        "",
        "--component_spec",
        f"{component_spec_string}",
    ]

    # # Instantiate and run component
    component = MyComponent.from_args()
    component.run()
