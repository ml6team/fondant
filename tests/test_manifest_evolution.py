import json
from pathlib import Path

import pytest
import yaml

from fondant.component_spec import FondantComponentSpec
from fondant.manifest import Manifest


examples_path = Path(__file__).parent / "example_specs/evolution_examples"


@pytest.fixture
def input_manifest():
    with open(examples_path / "input_manifest.json") as f:
        return json.load(f)


def examples():
    """Returns examples as tuples of component and expected output_manifest"""
    for directory in (f for f in examples_path.iterdir() if f.is_dir()):
        with open(directory / "component.yaml") as c, open(
            directory / "output_manifest.json"
        ) as o:
            yield yaml.safe_load(c), json.load(o)


@pytest.mark.parametrize("component_spec, output_manifest", list(examples()))
def test_evolution(input_manifest, component_spec, output_manifest):
    manifest = Manifest(input_manifest)
    component_spec = FondantComponentSpec(component_spec)
    evolved_manifest = manifest.evolve(component_spec=component_spec)

    assert evolved_manifest._specification == output_manifest
