"""Test scripts for import module functionality."""
import importlib.metadata
from unittest import mock
from unittest.mock import patch

import pytest

from fondant.import_utils import (
    is_datasets_available,
    is_kfp_available,
    is_package_available,
    is_pandas_available,
)


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
    """Test function for is_package_available()."""
    if isinstance(expected_result, bool):
        assert is_package_available(package_name, import_error_msg) == expected_result
    else:
        with expected_result:
            is_package_available(package_name, import_error_msg)


@patch("fondant.import_utils.is_package_available", return_value=True)
def test_available_packages(mock_is_package_available):
    """Test that is_datasets_available returns True when 'datasets' is available."""
    assert is_datasets_available() is True
    assert is_pandas_available() is True
    assert is_kfp_available() is True


@mock.patch(
    "importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError
)
def test_unavailable_packages(mock_importlib_metadata_version):
    """Test that is_datasets_available returns False when 'datasets' is not available."""
    with pytest.raises(ModuleNotFoundError):
        is_datasets_available()

    with pytest.raises(ModuleNotFoundError):
        is_pandas_available()

    with pytest.raises(ModuleNotFoundError):
        is_kfp_available()
