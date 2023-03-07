"""
Test scripts for io functionalities
"""
import pytest

from express_components.helpers.io_helpers import get_file_name, get_file_extension


@pytest.mark.parametrize(
    "file_uri, return_extension, expected_result",
    [
        ("gs://bucket/path/to/file", False, "file"),
        ("gs://bucket/path/to/file.txt", True, "file.txt"),
        ("gs://bucket/path/to/file.name.with.dots.csv", False, "file.name.with.dots"),
        ("gs://bucket/path/to/file.name.with.dots.csv", True, "file.name.with.dots.csv"),
    ],
)
def test_get_file_name(file_uri, return_extension, expected_result):
    assert get_file_name(file_uri, return_extension) == expected_result


@pytest.mark.parametrize(
    "file_name, expected_result",
    [
        ("file.jpg", "jpg"),
        ("file", ""),
        ("file.test.jpg", "jpg"),
        ("file/test.jpg", "jpg"),
    ],
)
def test_file_extension(file_name, expected_result):
    assert get_file_extension(file_name) == expected_result
