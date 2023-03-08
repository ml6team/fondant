"""
Test scripts for gcp storage functionalities
"""
import pytest

from express.storage_interface import DecodedBlobPath
from express.gcp_storage import StorageHandler


@pytest.mark.parametrize("fully_qualified_blob_path, expected_result", [

    ("gs://my-bucket/my-file.txt",
     DecodedBlobPath(bucket_name="my-bucket", blob_path="my-file.txt")),

    ("gs://my-bucket/my-folder/my-file.txt",
     DecodedBlobPath(bucket_name="my-bucket", blob_path="my-folder/my-file.txt")),

    ("gs://bucket/path/to/my-file.test.txt",
     DecodedBlobPath(bucket_name="bucket", blob_path="path/to/my-file.test.txt"))
])
def test_decode_blob_path(fully_qualified_blob_path, expected_result):
    handler = StorageHandler()
    assert handler.decode_blob_path(fully_qualified_blob_path) == expected_result
