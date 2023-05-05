import json
import pkgutil
from pathlib import Path

import pytest
from fondant.exceptions import InvalidManifest
from fondant.manifest import Subset, Manifest, Type


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
    """Test that the manifest is validated correctly on instantiation"""
    Manifest(valid_manifest)
    with pytest.raises(InvalidManifest):
        Manifest(invalid_manifest)


def test_subset_init():
    """Test initializing a subset"""
    subset_spec = {
        "location": "/ABC/123/images",
        "fields": {
            "data": {
                "type": "binary",
            },
        },
    }
    subset = Subset(specification=subset_spec, base_path="/tmp")
    assert subset.location == "/tmp/ABC/123/images"
    assert (
        subset.__repr__()
        == "Subset({'location': '/ABC/123/images', 'fields': {'data': {'type': 'binary'}}})"
    )


def test_subset_fields():
    """Test manipulating subset fields"""
    subset_spec = {
        "location": "/ABC/123/images",
        "fields": {
            "data": {
                "type": "binary",
            },
        },
    }
    subset = Subset(specification=subset_spec, base_path="/tmp")

    # add a field
    subset.add_field(name="data2", type_=Type.binary)
    assert "data2" in subset.fields

    # add a duplicate field
    with pytest.raises(ValueError):
        subset.add_field(name="data2", type_=Type.binary)

    # add a duplicate field but overwrite
    subset.add_field(name="data2", type_=Type.utf8, overwrite=True)
    assert subset.fields["data2"].type == "utf8"

    # remove a field
    subset.remove_field(name="data2")
    assert "data2" not in subset.fields


def test_set_base_path(valid_manifest):
    """Test altering the base path in the manifest"""
    manifest = Manifest(valid_manifest)
    tmp_path = "/tmp/base_path"
    manifest.base_path = tmp_path

    assert manifest.base_path == tmp_path
    assert manifest._specification["metadata"]["base_path"] == tmp_path


def test_from_to_file(valid_manifest):
    """Test reading from and writing to file"""
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
    - Dynamic properties should be accessible by lookup
    """
    manifest = Manifest(valid_manifest)

    assert manifest.metadata == valid_manifest["metadata"]
    assert manifest.index.location == "gs://bucket/index"
    assert manifest.subsets["images"].location == "gs://bucket/images"
    assert manifest.subsets["images"].fields["data"].type == "binary"


def test_manifest_creation():
    """Test the stepwise creation of a manifest via the Manifest class"""
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


def test_manifest_repr():
    manifest = Manifest.create(base_path="/", run_id="A", component_id="1")
    assert (
        manifest.__repr__()
        == "Manifest({'metadata': {'base_path': '/', 'run_id': 'A', 'component_id': '1'}, 'index': {'location': '/A/1/index'}, 'subsets': {}})"
    )


def test_manifest_alteration(valid_manifest):
    """Test alteration functionalities of a manifest via the Manifest class"""
    manifest = Manifest(valid_manifest)

    # test adding a subset
    manifest.add_subset("images2", [("width", Type.int32), ("height", Type.int32)])
    assert "images2" in manifest.subsets

    # test adding a duplicate subset
    with pytest.raises(ValueError):
        manifest.add_subset("images2", [("width", Type.int32), ("height", Type.int32)])

    # test removing a subset
    manifest.remove_subset("images2")
    assert "images2" not in manifest.subsets

    # test removing a nonexistant subset
    with pytest.raises(ValueError):
        manifest.remove_subset("pictures")


def test_manifest_copy_and_adapt(valid_manifest):
    """Test that a manifest can be copied and adapted without changing the original."""
    manifest = Manifest(valid_manifest)
    new_manifest = manifest.copy()
    new_manifest.remove_subset("images")
    assert manifest._specification == valid_manifest
    assert new_manifest._specification != valid_manifest


def test_no_validate_schema(monkeypatch, valid_manifest):
    monkeypatch.setattr(pkgutil, "get_data", lambda package, resource: None)
    with pytest.raises(FileNotFoundError):
        manifest = Manifest(valid_manifest)
