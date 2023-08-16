import json
from pathlib import Path

import pytest
import yaml
from fondant.component_spec import ComponentSpec
from fondant.manifest import Manifest

examples_path = Path(__file__).parent / "example_specs/evolution_examples"
examples_path_mapping = (
    Path(__file__).parent / "example_specs/evolution_examples_mapping"
)


@pytest.fixture()
def input_manifest():
    with open(examples_path / "input_manifest.json") as f:
        return json.load(f)


@pytest.fixture()
def input_manifest_mapping():
    with open(examples_path_mapping / "input_manifest.json") as f:
        return json.load(f)


def examples():
    """Returns examples as tuples of component and expected output_manifest."""
    for directory in (f for f in examples_path.iterdir() if f.is_dir()):
        with open(directory / "component.yaml") as c, open(
            directory / "output_manifest.json",
        ) as o:
            yield yaml.safe_load(c), json.load(o)


def examples_mapping():
    """Returns examples as tuples of mapping dicts, component and expected output_manifest."""
    spec_mapping = {
        "1": {
            "pictures_array": "images_data",
        },
        "2": {
            "text_lines": "texts_data",
        },
    }

    for directory in (f for f in examples_path_mapping.iterdir() if f.is_dir()):
        with open(directory / "component.yaml") as c, open(
            directory / "output_manifest.json",
        ) as o:
            yield spec_mapping[directory.name], yaml.safe_load(c), json.load(o)


@pytest.mark.parametrize(("component_spec", "output_manifest"), list(examples()))
def test_evolution(input_manifest, component_spec, output_manifest):
    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(component_spec)
    evolved_manifest = manifest.evolve(component_spec=component_spec)

    assert evolved_manifest._specification == output_manifest


@pytest.mark.parametrize(
    ("spec_mapping", "component_spec", "output_manifest"),
    list(examples_mapping()),
)
def test_evolution_mapping(
    input_manifest_mapping,
    spec_mapping,
    component_spec,
    output_manifest,
):
    manifest = Manifest(input_manifest_mapping)
    component_spec = ComponentSpec(
        component_spec,
        spec_mapping=spec_mapping,
    )
    evolved_manifest = manifest.evolve(component_spec=component_spec)

    assert evolved_manifest._specification == output_manifest


def test_component_spec_location_update():
    with open(examples_path / "input_manifest.json") as f:
        input_manifest = json.load(f)

    with open(examples_path / "7/component.yaml") as f:
        specification = yaml.safe_load(f)

    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(specification)
    evolved_manifest = manifest.evolve(component_spec=component_spec)

    assert (
        evolved_manifest._specification["subsets"]["images"]["location"]
        == "/images/12345/example_component"
    )
