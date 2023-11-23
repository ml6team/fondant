import json
import pkgutil
from collections import OrderedDict
from pathlib import Path

import pytest
from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidManifest
from fondant.core.manifest import Field, Manifest, Type

manifest_path = Path(__file__).parent / "examples" / "manifests"
component_specs_path = Path(__file__).parent / "examples" / "component_specs"


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
    assert manifest.index.location == "/component1"
    assert manifest.fields["images"].location == "/component1"
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

    manifest.add_or_update_field(Field(name="width", type=Type("int32")))
    manifest.add_or_update_field(Field(name="height", type=Type("int32")))
    manifest.add_or_update_field(Field(name="data", type=Type("binary")))

    assert manifest._specification == {
        "metadata": {
            "pipeline_name": pipeline_name,
            "base_path": base_path,
            "run_id": run_id,
            "component_id": component_id,
            "cache_key": cache_key,
        },
        "index": {"location": f"/{component_id}"},
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
        " 'index': {'location': '/1'}, 'fields': {}})"
    )


def test_manifest_alteration(valid_manifest):
    """Test alteration functionalities of a manifest via the Manifest class."""
    manifest = Manifest(valid_manifest)

    # test adding a subset
    manifest.add_or_update_field(Field(name="width2", type=Type("int32")))
    manifest.add_or_update_field(Field(name="height2", type=Type("int32")))

    assert "width2" in manifest.fields
    assert "height2" in manifest.fields

    # test adding a duplicate subset
    with pytest.raises(ValueError, match="A field with name width2 already exists"):
        manifest.add_or_update_field(Field(name="width2", type=Type("int32")))

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


def test_evolve_manifest():
    """Test that the fields are evolved as expected."""
    run_id = "A"
    spec = ComponentSpec.from_file(component_specs_path / "valid_component.yaml")
    input_manifest = Manifest.create(
        pipeline_name="NAME",
        base_path="/base_path",
        run_id=run_id,
        component_id="component_1",
        cache_key="42",
    )

    output_manifest = input_manifest.evolve(component_spec=spec, run_id=run_id)

    assert output_manifest.base_path == input_manifest.base_path
    assert output_manifest.run_id == run_id
    assert output_manifest.index.location == "/" + spec.component_folder_name
    assert output_manifest.fields["captions"].type.name == "string"


def test_fields():
    """Test that the fields can added and updated as expected."""
    run_id = "A"
    manifest = Manifest.create(
        pipeline_name="NAME",
        base_path="/base_path",
        run_id=run_id,
        component_id="component_1",
        cache_key="42",
    )

    # add a field
    manifest.add_or_update_field(Field(name="field_1", type=Type("int32")))
    assert "field_1" in manifest.fields

    # add a duplicate field, but overwrite (update)
    manifest.add_or_update_field(
        Field(name="field_1", type=Type("string")),
        overwrite=True,
    )
    assert manifest.fields["field_1"].type.name == "string"

    # add duplicate field
    with pytest.raises(
        ValueError,
        match="A field with name field_1 already exists. Set overwrite to true, "
        "if you want to update the field.",
    ):
        manifest.add_or_update_field(
            Field(name="field_1", type=Type("string")),
            overwrite=False,
        )

    # delete a field
    manifest.remove_field(name="field_1")
    assert "field_1" not in manifest.fields


def test_field_mapping(valid_manifest):
    """Test field mapping generation."""
    manifest = Manifest(valid_manifest)
    manifest.add_or_update_field(Field(name="index", location="component2"))
    field_mapping = manifest.field_mapping
    assert field_mapping == OrderedDict(
        {
            "gs://bucket/test_pipeline/test_pipeline_12345/component2": [
                "id",
                "height",
                "width",
            ],
            "gs://bucket/test_pipeline/test_pipeline_12345/component1": ["images"],
            "gs://bucket/test_pipeline/test_pipeline_12345/component3": ["caption"],
        },
    )
