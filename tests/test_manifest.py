import json
import pytest
from express.exceptions import InvalidManifest
from express.manifest import Manifest

VALID_MANIFEST = {
    "metadata": {
        "base_path": "gs://bucket"
    },
    "index": {
        "location": "/index"
    },
    "subsets": {
        "images": {
            "location": "/images",
            "fields": {
                "data": {
                    "type": "bytes"
                },
                "height": {
                    "type": "int"
                },
                "width": {
                    "type": "int"
                }
            }
        },
        "captions": {
            "location": "/captions",
            "fields": {
                "data": {
                    "type": "bytes"
                }
            }
        }
    }
}

INVALID_MANIFEST = {
    "metadata": {
        "base_path": "gs://bucket"
    },
    "index": {
        "location": "/index"
    },
    "subsets": {
        "images": {
            "location": "/images",
            "fields": []  # Should be an object
        }
    }
}


def test_manifest_validation():
    """Test that the manifest is validated correctly on instantiation"""
    Manifest(VALID_MANIFEST)
    with pytest.raises(InvalidManifest):
        Manifest(INVALID_MANIFEST)


def test_from_to_file():
    """Test reading from and writing to file"""
    tmp_path = "/tmp/manifest.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(VALID_MANIFEST, f)

    manifest = Manifest.from_file(tmp_path)
    assert manifest.metadata == VALID_MANIFEST["metadata"]

    manifest.to_file(tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        assert json.load(f) == VALID_MANIFEST


def test_attribute_access():
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup
    """
    manifest = Manifest(VALID_MANIFEST)

    assert manifest.metadata == VALID_MANIFEST["metadata"]
    assert manifest.index.location == "gs://bucket/index"
    assert manifest.subsets["images"].location == "gs://bucket/images"
    assert manifest.subsets["images"].fields["data"].type == "bytes"
