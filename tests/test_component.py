"""Fondant component tests"""

import argparse
from unittest import mock 

from fondant.component import FondantLoadComponent
from fondant.component_spec import FondantComponentSpec


class LoadFromHubComponent(FondantLoadComponent):
    def __init__(self):
        self.spec = FondantComponentSpec.from_file("tests/example_specs/valid_component/fondant_component.yaml")
        self.args = self._add_and_parse_args()
    
    def load(self):
        return -1


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(input_manifest_path=".", output_manifest_path="."))
def test_component(mock_args):
    """Test that the manifest is validated correctly on instantiation"""
    component = LoadFromHubComponent()
    assert "input_manifest_path" in component._get_component_arguments()
    assert "output_manifest_path" in component._get_component_arguments()
    assert argparse.Namespace(input_manifest_path=".", output_manifest_path=".") == component._add_and_parse_args()
