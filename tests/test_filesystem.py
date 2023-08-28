import re

import pytest
from fondant.filesystem import get_filesystem


def test_valid_filesystem():
    """Test that a data type specified with the Type class matches the expected pyarrow schema."""

    def get_instance_class_name(instance):
        return instance.__class__.__name__

    assert get_instance_class_name(get_filesystem("/home/foo/bar")) == "LocalFileSystem"
    assert get_instance_class_name(get_filesystem("gs://foo/bar")) == "GCSFileSystem"
    assert get_instance_class_name(get_filesystem("s3://foo/bar")) == "S3FileSystem"


def test_invalid_filesystem():
    """Test that a data type specified with the Type class matches the expected pyarrow schema."""
    expected_msg = "Protocol not known: invalid_protocol"
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        get_filesystem("invalid_protocol://")
