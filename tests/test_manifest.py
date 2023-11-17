import json
import pkgutil
from pathlib import Path

import pytest
from fondant.core.exceptions import InvalidManifest
from fondant.core.manifest import Field, Index, Manifest, Subset, Type

manifest_path = Path(__file__).parent / "example_specs/manifests"


@pytest.fixture()
def valid_manifest():
    with open(manifest_path / "valid_manifest.json") as f:
        return json.load(f)


@pytest.fixture()
def invalid_manifest():
    with open(manifest_path / "invalid_manifest.json") as f:
        return json.load(f)


def test_manifest_validation(valid_manifest, invalid_manifest):
    """Test that the manifest is validated correctly on instantiation."""
    Manifest(valid_manifest)
    with pytest.raises(InvalidManifest):
        Manifest(invalid_manifest)


def test_subset_init():
    """Test initializing a subset."""
    subset_spec = {
        "location": "/images/ABC/123",
        "fields": {
            "data": {
                "type": "binary",
            },
        },
    }
    subset = Subset(specification=subset_spec, base_path="/tmp")
    assert subset.location == "/tmp/images/ABC/123"
    assert (
        subset.__repr__()
        == "Subset({'location': '/images/ABC/123', 'fields': {'data': {'type': 'binary'}}})"
    )


def test_subset_fields():
    """Test manipulating subset fields."""
    subset_spec = {
        "location": "/images/ABC/123",
        "fields": {
            "data": {
                "type": "binary",
            },
        },
    }
    subset = Subset(specification=subset_spec, base_path="/tmp")

    # add a field
    subset.add_field(name="data2", type_=Type("binary"))
    assert "data2" in subset.fields

    # add a duplicate field
    with pytest.raises(ValueError, match="A field with name data2 already exists"):
        subset.add_field(name="data2", type_=Type("binary"))

    # add a duplicate field but overwrite
    subset.add_field(name="data2", type_=Type("string"), overwrite=True)
    assert subset.fields["data2"].type == Type("string")

    # remove a field
    subset.remove_field(name="data2")
    assert "data2" not in subset.fields


def test_set_base_path(valid_manifest):
    """Test altering the base path in the manifest."""
    manifest = Manifest(valid_manifest)
    tmp_path = "/tmp/base_path"
    manifest.update_metadata(key="base_path", value=tmp_path)

    assert manifest.base_path == tmp_path
    assert manifest._specification["metadata"]["base_path"] == tmp_path


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
    assert manifest.index.location == "gs://bucket/component1"
    assert manifest.fields["images"].location == "gs://bucket/component1"
    assert manifest.fields["images"].type == Type("binary")


def test_manifest_creation():
    """Test the stepwise creation of a manifest via the Manifest class."""
    base_path = "gs://bucket"
    run_id = "run_id"
    pipeline_name = "pipeline_name"
    component_id = "component_id"
    cache_key = "42"

    manifest = Manifest.create(
        pipeline_name=pipeline_name,
        base_path=base_path,
        run_id=run_id,
        component_id=component_id,
        cache_key=cache_key,
    )

    manifest.add_fields(
        [
            Field(name="width", type=Type("int32")),
            Field(name="height", type=Type("int32")),
        ]
    )
    manifest.add_field(Field(name="data", type=Type("binary")))

    assert manifest._specification == {
        "metadata": {
            "pipeline_name": pipeline_name,
            "base_path": base_path,
            "run_id": run_id,
            "component_id": component_id,
            "cache_key": cache_key,
        },
        "index": {"location": f"/{pipeline_name}/{run_id}/{component_id}/index"},
        "fields": {
            "width": {
                "type": "int32",
                "location": f"/{component_id}",
            },
            "height": {
                "type": "int32",
                "location": f"/{component_id}",
            },
            "data": {
                "type": "binary",
                "location": f"/{component_id}",
            },
        },
    }


def test_manifest_repr():
    manifest = Manifest.create(
        pipeline_name="NAME",
        base_path="/",
        run_id="A",
        component_id="1",
        cache_key="42",
    )
    assert (
        manifest.__repr__()
        == "Manifest({'metadata': {'base_path': '/', 'pipeline_name': 'NAME', 'run_id': 'A',"
        " 'component_id': '1', 'cache_key': '42'},"
        " 'index': {'location': '/NAME/A/1/index'}, 'fields': {}})"
    )


def test_manifest_alteration(valid_manifest):
    """Test alteration functionalities of a manifest via the Manifest class."""
    manifest = Manifest(valid_manifest)

    # test adding a subset
    manifest.add_fields(
        [
            Field(name="width2", type=Type("int32")),
            Field(name="height2", type=Type("int32")),
        ],
    )

    assert "width2" in manifest.fields
    assert "height2" in manifest.fields

    # test adding a duplicate subset
    with pytest.raises(ValueError, match="A field with name width2 already exists"):
        manifest.add_fields(
            [
                Field(name="width2", type=Type("int32")),
            ],
        )

    # test removing a subset
    manifest.remove_field("width2")
    assert "images2" not in manifest.fields

    # test removing a nonexistant subset
    with pytest.raises(ValueError, match="Field pictures not found in specification"):
        manifest.remove_field("pictures")


def test_manifest_copy_and_adapt(valid_manifest):
    """Test that a manifest can be copied and adapted without changing the original."""
    manifest = Manifest(valid_manifest)
    new_manifest = manifest.copy()
    new_manifest.remove_field("images")
    assert manifest._specification == valid_manifest
    assert new_manifest._specification != valid_manifest


def test_no_validate_schema(monkeypatch, valid_manifest):
    monkeypatch.setattr(pkgutil, "get_data", lambda package, resource: None)
    with pytest.raises(FileNotFoundError):
        Manifest(valid_manifest)


def test_index_fields():
    """Test that the fields property of Index returns the expected fields."""
    subset_spec = {
        "location": "/images/ABC/123",
        "fields": {
            "data": {
                "type": "binary",
            },
        },
    }

    index = Index(specification=subset_spec, base_path="/tmp")

    expected_fields = {
        "id": Field(name="id", type=Type("string")),
        "source": Field(name="source", type=Type("string")),
    }

    assert index.fields == expected_fields
