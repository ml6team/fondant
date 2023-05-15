"""Fondant component tests."""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import pytest

from fondant.component import LoadComponent, WriteComponent
from fondant.data_io import DaskDataLoader

components_path = Path(__file__).parent / "example_specs/components"
component_specs_path = Path(__file__).parent / "example_specs/component_specs"


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
    component_args = component._get_component_arguments()
    assert "input_manifest_path" in component_args
    assert "output_manifest_path" in component_args
    assert (
        argparse.Namespace(
            input_manifest_path=".",
            output_manifest_path="result.parquet",
            metadata=metadata,
        )
        == component._add_and_parse_args()
    )

    # test custom args
    assert list(component.spec.args) == ["storage_args"]

    # test manifest
    initial_manifest = component._load_or_create_manifest()
    assert initial_manifest.metadata == {
        "base_path": ".",
        "run_id": "200",
        "component_id": "example_component",
    }


def test_transform_kwargs(monkeypatch):
    """Test that arguments are passed correctly to `Component.transform` method."""

    class EarlyStopException(Exception):
        """Used to stop execution early instead of mocking all later functionality."""

    # Mock `Dataset.load_dataframe` so no actual data is loaded
    def mocked_load_dataframe(self, spec):
        return dd.from_dict({"a": [1, 2, 3]}, npartitions=1)

    monkeypatch.setattr(DaskDataLoader, "load_dataframe", mocked_load_dataframe)

    # Define paths to specs to instantiate component
    arguments_dir = components_path / "arguments"
    component_spec = arguments_dir / "component.yaml"
    input_manifest = arguments_dir / "input_manifest.json"

    # Implemented Component class
    class MyComponent(WriteComponent):
        def transform(self, dataframe, *, flag, value):
            assert flag == "success"
            assert value == 1
            raise EarlyStopException()

    # Mock CLI arguments
    sys.argv = [
        "",
        "--input_manifest_path",
        str(input_manifest),
        "--flag",
        "success",
        "--value",
        "1",
        "--output_manifest_path",
        "",
    ]

    # Instantiate and run component
    component = MyComponent.from_file(component_spec)
    with pytest.raises(EarlyStopException):
        component.run()
