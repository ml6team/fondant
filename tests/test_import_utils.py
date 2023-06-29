"""Test scripts for import module functionality."""
import importlib.metadata
from unittest import mock

import pytest

from fondant.import_utils import (
    is_datasets_available,
    is_kfp_available,
    is_package_available,
    is_pandas_available,
)


@pytest.mark.parametrize(
    ("package_name", "import_error_msg", "expected_result"),
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
    """Test function for is_package_available()."""
    if isinstance(expected_result, bool):
        assert is_package_available(package_name, import_error_msg) == expected_result
    else:
        with expected_result:
            is_package_available(package_name, import_error_msg)


@mock.patch("importlib.util.find_spec", return_value="package")
@mock.patch("importlib.metadata.version", return_value="0.1.0")
def test_available_packages(importlib_util_find_spec, importlib_metadata_version):
    """Test that is_datasets_available is not False when the packages are available."""
    assert is_datasets_available() is not False
    assert is_pandas_available() is not False
    assert is_kfp_available() is not False


@mock.patch(
    "importlib.metadata.version",
    side_effect=importlib.metadata.PackageNotFoundError,
)
def test_unavailable_packages(mock_importlib_metadata_version):
    """Test that is_datasets_available returns False when 'datasets' is not available."""
    with pytest.raises(ModuleNotFoundError):
        is_datasets_available()

    with pytest.raises(ModuleNotFoundError):
        is_pandas_available()

    with pytest.raises(ModuleNotFoundError):
        is_kfp_available()
