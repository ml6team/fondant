import pandas as pd

from src.main import NormalizeTextComponent


def test_transform_custom_componen_test():
    """Test components transform method."""
    user_arguments = {
        "remove_additional_whitespaces": True,
        "apply_nfc": True,
        "normalize_lines": True,
        "do_lowercase": True,
        "remove_punctuation": True,
    }
    component = NormalizeTextComponent(**user_arguments)

    input_dataframe = pd.DataFrame(
        [
            "\u0043\u0327 something",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Nulla facilisi. Sed eu nulla sit amet enim scelerisque dapibus.",
        ],
        columns=["text"],
    )

    expected_output = pd.DataFrame(
        [
            "\u00e7 something",
            "lorem ipsum dolor sit amet consectetur adipiscing elit",
            "nulla facilisi sed eu nulla sit amet enim scelerisque dapibus",
        ],
        columns=["text"],
    )

    output_dataframe = component.transform(input_dataframe)

    pd.testing.assert_frame_equal(
        left=expected_output,
        right=output_dataframe,
        check_dtype=False,
    )
