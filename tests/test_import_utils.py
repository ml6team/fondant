"""
Test scripts for import module functionality
"""
import pytest

from fondant.import_utils import is_package_available


@pytest.mark.parametrize(
    "package_name, import_error_msg, expected_result",
    [
        ("jsonschema", "jsonschema package is not installed.", True),
        (
            "nonexistentpackage",
            "This package does not exist.",
            pytest.raises(ModuleNotFoundError),
        ),
    ],
)
def test_is_package_available(package_name, import_error_msg, expected_result):
    """
    Test function for is_package_available().
    """
    if isinstance(expected_result, bool):
        assert is_package_available(package_name, import_error_msg) == expected_result
    else:
        with expected_result:
            is_package_available(package_name, import_error_msg)
