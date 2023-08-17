import json
from pathlib import Path

import pytest
import yaml
from fondant.component_spec import ColumnMapping, ComponentSpec
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
    column_mapping_list = {
        "1": [
            ColumnMapping(
                dataset_column="pictures_array",
                component_column="images_data",
            ),
        ],
        "2": [
            ColumnMapping(dataset_column="text_lines", component_column="texts_data"),
        ],
    }

    for directory in (f for f in examples_path_mapping.iterdir() if f.is_dir()):
        with open(directory / "component.yaml") as c, open(
            directory / "output_manifest.json",
        ) as o:
            yield column_mapping_list[directory.name], yaml.safe_load(c), json.load(o)


@pytest.mark.parametrize(("component_spec", "output_manifest"), list(examples()))
def test_evolution(input_manifest, component_spec, output_manifest):
    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(component_spec)
    evolved_manifest = manifest.evolve(component_spec=component_spec)

    assert evolved_manifest._specification == output_manifest


@pytest.mark.parametrize(
    ("column_mapping_list", "component_spec", "output_manifest"),
    list(examples_mapping()),
)
def test_evolution_mapping(
    input_manifest_mapping,
    column_mapping_list,
    component_spec,
    output_manifest,
):
    column_mapping = ColumnMapping.list_to_dict(column_mapping_list)
    manifest = Manifest(input_manifest_mapping)
    component_spec = ComponentSpec(
        component_spec,
        column_mapping=column_mapping,
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
