from typing import Any, Dict

import pandas as pd
import pytest
from fondant.testing_utils import execute_pandas_transform_component

from src.main import TextNormalizationComponent


def test_transform_custom_componen_test():
    """Test components transform method.
    Option 1: handling the test case is up to the users.
    """
    user_arguments = {
        "remove_additional_whitespaces": True,
        "apply_nfc": True,
        "remove_bad_patterns": True,
        "do_lowercase": True,
        "remove_punctuation": True,
    }
    component = TextNormalizationComponent(**user_arguments)

    input_dataframe = pd.DataFrame([
        "\u0043\u0327 something",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Nulla facilisi. Sed eu nulla sit amet enim scelerisque dapibus.",
    ], columns=[("text", "data")])

    expected_output = pd.DataFrame([
        "\u00e7 something",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "nulla facilisi sed eu nulla sit amet enim scelerisque dapibus",
    ], columns=[("text", "data")])

    output_dataframe = component.transform(input_dataframe)

    pd.testing.assert_frame_equal(
        left=expected_output,
        right=output_dataframe,
        check_dtype=False,
    )


def test_transform_helper_methods():
    """Test components transform method.
    Option 2: using helper method provided by fondant.
    """
    user_arguments = {
        "remove_additional_whitespaces": True,
        "apply_nfc": True,
        "remove_bad_patterns": True,
        "do_lowercase": True,
        "remove_punctuation": True,
    }
    component = TextNormalizationComponent(**user_arguments)

    input_dataframe = pd.DataFrame([
        "\u0043\u0327 something",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Nulla facilisi. Sed eu nulla sit amet enim scelerisque dapibus.",
    ], columns=[("text", "data")])

    expected_output = pd.DataFrame([
        "\u00e7 something",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "nulla facilisi sed eu nulla sit amet enim scelerisque dapibus",
    ], columns=[("text", "data")])

    execute_pandas_transform_component(component, input_dataframe, expected_output)


data = [
    # first scenario
    {
        "user_arguments": {
            "remove_additional_whitespaces": True,
            "apply_nfc": True,
            "remove_bad_patterns": True,
            "do_lowercase": True,
            "remove_punctuation": True,
        },
        "input_dataframe": pd.DataFrame([
            "\u0043\u0327 something",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Nulla facilisi. Sed eu nulla sit amet enim scelerisque dapibus.",
        ], columns=[("text", "data")]),
        "output_dataframe": pd.DataFrame([
            "\u00e7 something",
            "lorem ipsum dolor sit amet consectetur adipiscing elit",
            "nulla facilisi sed eu nulla sit amet enim scelerisque dapibus",
        ], columns=[("text", "data")]),
    },

    # second scenario
    {
        "user_arguments": {
            "remove_additional_whitespaces": True,
            "apply_nfc": True,
            "remove_bad_patterns": True,
            "do_lowercase": False,
            "remove_punctuation": True,
        },
        "input_dataframe": pd.DataFrame([
            "Nulla facilisi. Sed eu nulla sit amet enim scelerisque dapibus.",
        ], columns=[("text", "data")]),
        "output_dataframe": pd.DataFrame([
            "Nulla facilisi Sed eu nulla sit amet enim scelerisque dapibus",
        ], columns=[("text", "data")]),
    },
]


@pytest.mark.parametrize(
    "scenario",
    data,
)
def test_transform_helper_methods_parametrized(scenario: Dict[str, Any]):
    """Option 3: Only defining parametrized scenarios. Usage of helper provided by fondant."""
    component = TextNormalizationComponent(**scenario["user_arguments"])
    execute_pandas_transform_component(component,
                                       scenario["input_dataframe"],
                                       scenario["output_dataframe"])
