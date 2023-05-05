import json
from pathlib import Path

import pytest

from fondant.exceptions import InvalidManifest
from fondant.manifest import Manifest, Type

manifest_path = Path(__file__).parent / "example_specs/manifests"


@pytest.fixture
def valid_manifest():
    with open(manifest_path / "valid_manifest.json") as f:
        return json.load(f)


@pytest.fixture
def invalid_manifest():
    with open(manifest_path / "invalid_manifest.json") as f:
        return json.load(f)


def test_manifest_validation(valid_manifest, invalid_manifest):
    """Test that the manifest is validated correctly on instantiation."""
    Manifest(valid_manifest)
    with pytest.raises(InvalidManifest):
        Manifest(invalid_manifest)


def test_from_to_file(valid_manifest):
    """Test reading from and writing to file."""
    tmp_path = "/tmp/manifest.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(valid_manifest, f)

    manifest = Manifest.from_file(tmp_path)
    assert manifest.metadata == valid_manifest["metadata"]

    manifest.to_file(tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        assert json.load(f) == valid_manifest


def test_attribute_access(valid_manifest):
    """
    Test that attributes can be accessed as expected:
    - Fixed properties should be accessible as an attribute
    - Dynamic properties should be accessible by lookup.
    """
    manifest = Manifest(valid_manifest)

    assert manifest.metadata == valid_manifest["metadata"]
    assert manifest.index.location == "gs://bucket/index"
    assert manifest.subsets["images"].location == "gs://bucket/images"
    assert manifest.subsets["images"].fields["data"].type == "binary"


def test_manifest_creation():
    """Test the stepwise creation of a manifest via the Manifest class."""
    base_path = "gs://bucket"
    run_id = "run_id"
    component_id = "component_id"

    manifest = Manifest.create(
        base_path=base_path, run_id=run_id, component_id=component_id
    )
    manifest.add_subset("images", [("width", Type.int32), ("height", Type.int32)])
    manifest.subsets["images"].add_field("data", Type.binary)

    assert manifest._specification == {
        "metadata": {
            "base_path": base_path,
            "run_id": run_id,
            "component_id": component_id,
        },
        "index": {"location": f"/{run_id}/{component_id}/index"},
        "subsets": {
            "images": {
                "location": f"/{run_id}/{component_id}/images",
                "fields": {
                    "width": {
                        "type": "int32",
                    },
                    "height": {
                        "type": "int32",
                    },
                    "data": {
                        "type": "binary",
                    },
                },
            }
        },
    }


def test_manifest_copy_and_adapt(valid_manifest):
    """Test that a manifest can be copied and adapted without changing the original."""
    manifest = Manifest(valid_manifest)
    new_manifest = manifest.copy()
    new_manifest.remove_subset("images")
    assert manifest._specification == valid_manifest
    assert new_manifest._specification != valid_manifest
