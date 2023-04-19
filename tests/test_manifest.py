import json
import pytest
from express.exceptions import InvalidManifest
from express.manifest import Manifest, Type

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
                    "type": "binary"
                },
                "height": {
                    "type": "int32"
                },
                "width": {
                    "type": "int32"
                }
            }
        },
        "captions": {
            "location": "/captions",
            "fields": {
                "data": {
                    "type": "binary"
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
    assert manifest.subsets["images"].fields["data"].type == "binary"


def test_manifest_creation():
    """Test the stepwise creation of a manifest via the Manifest class"""
    base_path = "gs://bucket"
    run_id = "run_id"
    component_id = "component_id"

    manifest = Manifest.create(base_path=base_path, run_id=run_id, component_id=component_id)
    manifest.add_subset("images", [("width", "int32"), ("height", "int32")])
    manifest.subsets["images"].add_field("data", Type.binary)

    assert manifest._specification == {
        "metadata": {
            "base_path": base_path,
            "run_id": run_id,
            "component_id": component_id,
        },
        "index": {
            "location": f"/index/{run_id}/{component_id}"
        },
        "subsets": {
            "images": {
                "location": f"/images/{run_id}/{component_id}",
                "fields": {
                    "width": {
                        "type": "int32",
                    },
                    "height": {
                        "type": "int32",
                    },
                    "data": {
                        "type": "binary",
                    }
                }
            }
        }
    }


def test_manifest_copy_and_adapt():
    """Test that a manifest can be copied and adapted without changing the original."""
    manifest = Manifest(VALID_MANIFEST)
    new_manifest = manifest.copy()
    new_manifest.remove_subset("images")
    assert manifest._specification == VALID_MANIFEST
    assert new_manifest._specification != VALID_MANIFEST
