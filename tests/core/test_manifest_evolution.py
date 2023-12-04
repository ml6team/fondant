import json
from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from fondant.core.component_spec import ComponentSpec
from fondant.core.manifest import Manifest
from fondant.core.schema import Field, Type

examples_path = Path(__file__).parent / "examples/evolution_examples"

TEST_MANIFESTS = {
    "1": {"produces": None},
    "2": {"produces": None},
    "3": {"produces": None},
    "4": {"produces": None},
    # Generic component
    "5": {
        "produces": {
            "embedding_data": Field("embedding_data", Type(pa.list_(pa.float32()))),
        },
    },
    # Non-generic component (output mapping)
    "6": {
        "produces": {
            "images_data": "images_array",
            "text_data": "text_string",
        },
    },
}


@pytest.fixture()
def input_manifest():
    with open(examples_path / "input_manifest.json") as f:
        return json.load(f)


def examples():
    """Returns examples as tuples of component and expected output_manifest."""
    for directory in (f for f in examples_path.iterdir() if f.is_dir()):
        example_name = directory.stem
        produces = TEST_MANIFESTS[example_name]["produces"]
        with open(directory / "component.yaml") as c, open(
            directory / "output_manifest.json",
        ) as o:
            yield yaml.safe_load(c), json.load(o), produces


@pytest.mark.parametrize(
    ("component_spec", "output_manifest", "produces"),
    list(examples()),
)
def test_evolution(input_manifest, component_spec, output_manifest, produces):
    run_id = "custom_run_id"
    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(component_spec)
    evolved_manifest = manifest.evolve(
        component_spec=component_spec,
        run_id=run_id,
        produces=produces,
    )

    assert evolved_manifest._specification == output_manifest


def test_component_spec_location_update():
    with open(examples_path / "input_manifest.json") as f:
        input_manifest = json.load(f)

    with open(examples_path / "4/component.yaml") as f:
        specification = yaml.safe_load(f)

    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(specification)
    evolved_manifest = manifest.evolve(
        component_spec=component_spec,
    )

    assert evolved_manifest.index.location == "/" + component_spec.component_folder_name
