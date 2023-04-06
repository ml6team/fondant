"""
Test scripts for io functionalities
"""
import pytest
from typing import Any, List

from express.io import get_file_name, get_file_extension, create_subprocess_arguments, \
    get_path_from_url


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
    """Test get file name function"""
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
    """Test get file extension function"""
    assert get_file_extension(file_name) == expected_result


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        (["foo", "--bar"], {"baz": 123, "--qux": True}, ["--foo", "--bar", "-baz", "123", "--qux"]),
        ([], {"foo": "bar"}, ["-foo", "bar"]),
        ([], {}, [])
    ]
)
def test_create_subprocess_arguments(args: List[str], kwargs: dict[str, Any],
                                     expected: List[str]) -> None:
    """Test create subprocess argument function"""
    assert create_subprocess_arguments(args, kwargs) == expected


@pytest.mark.parametrize("url, expected_path", [
    ("gs://bucket/blob/image", "bucket/blob/image"),
    ("https://www.google.com/search?q=python", "www.google.com/search"),
    ("ftp://ftp.example.com/dir/file.txt", "ftp.example.com/dir/file.txt")
])
def test_get_path_from_url(url, expected_path):
    assert get_path_from_url(url) == expected_path
