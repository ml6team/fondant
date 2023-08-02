import json
import os
import typing as t
from glob import glob

import pandas as pd
import pytest
from fondant.component_spec import ComponentSpec

from components.text_normalization.src.main import TextNormalizationComponent


class MockedComponentSpec(ComponentSpec):
    """Just for mocking purpose. This component spec is not needed for unit testing."""

    def __init__(self, specification: t.Dict[str, t.Any]):
        pass


def load_fixtures(path="./fixtures"):
    test_configurations = []
    fixture_list = glob(path + "/*.json")
    for fixture in fixture_list:
        with open(fixture) as file:
            fixture_dict = json.load(file)

        fixture_name = os.path.splitext(fixture)[0]
        user_arguments = fixture_dict["user_arguments"]
        input_data = {
            tuple(key.split("_")): value for key, value in fixture_dict["input"].items()
        }
        expected_out = {
            tuple(key.split("_")): value
            for key, value in fixture_dict["output"].items()
        }

        test_configurations.append((fixture_name, user_arguments, input_data, expected_out))

    return test_configurations


@pytest.mark.parametrize(("fixture_name", "user_arguments", "input_data", "expected_output"),
                         load_fixtures())
def test_component(fixture_name, user_arguments, input_data, expected_output):
    """Test transform method of text normalization component."""
    print("Running test case based on: ", fixture_name)
    component = TextNormalizationComponent(MockedComponentSpec({}), **user_arguments)
    input_df = pd.DataFrame(input_data)
    transformed_output = component.transform(input_df)
    pd.testing.assert_frame_equal(pd.DataFrame(expected_output), transformed_output)
