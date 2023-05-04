"""Fondant component tests"""

import argparse
import json
import tempfile
from unittest import mock 

import dask.dataframe as dd

from fondant.component import FondantLoadComponent
from fondant.component_spec import FondantComponentSpec


class LoadFromHubComponent(FondantLoadComponent):
    def __init__(self):
        # overwrite to correctly point to the location of the yaml file
        self.spec = FondantComponentSpec.from_file("tests/example_specs/valid_component/fondant_component.yaml")
        self.args = self._add_and_parse_args()
    
    def load(self, args):
        data = {"id": [0,1], "source": ["cloud", "cloud"], "captions_data": ["hello world", "this is another caption"]}

        df = dd.DataFrame.from_dict(data, npartitions=1)
        
        return df


# we mock the argparse arguments in order to run the tests without having to pass arguments
metadata = json.dumps({"base_path": ".", "run_id": "200"})
@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(input_manifest_path=".", output_manifest_path="result.parquet", metadata=metadata))
def test_component(mock_args):
    """Test that the manifest is validated correctly on instantiation"""
    component = LoadFromHubComponent()
    
    # test component args
    component_args = component._get_component_arguments()
    assert "input_manifest_path" in component_args
    assert "output_manifest_path" in component_args
    assert argparse.Namespace(input_manifest_path=".", output_manifest_path="result.parquet", metadata=metadata) == component._add_and_parse_args()
    
    # test custom args
    component_spec_args = [arg.name for arg in component.spec.args]
    assert component_spec_args == ["storage_args"]

    # test manifest
    initial_manifest = component._load_or_create_manifest()
    assert initial_manifest.metadata == {'base_path': '.', 'run_id': '200', 'component_id': 'example_component'}

    # Create temporary directory for writing the subsets
    with tempfile.TemporaryDirectory(dir="."):
        component.run()

