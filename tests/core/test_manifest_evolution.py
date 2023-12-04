import json
from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from fondant.core.component_spec import ComponentSpec
from fondant.core.exceptions import InvalidPipelineDefinition
from fondant.core.manifest import Manifest
from fondant.core.schema import Field, Type

EXAMPLES_PATH = Path(__file__).parent / "examples/evolution_examples"

VALID_EXAMPLE = {
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

INVALID_EXAMPLES = {
    # Invalid data type in the produces mapping
    "1": {
        "produces": {
            "embedding_data": True,
        },
    },
    "3": {
        "produces": {
            "images_array": "numpy_array",
        },
    },
    # Generic component that has a string in the produces mapping
    "5": {
        "produces": {
            "images_data": "images_data",
        },
    },
    # Non-generic component that has a Field in the produces mapping
    "6": {
        "produces": {
            "embedding_data": Field("embedding_data", Type(pa.list_(pa.float32()))),
        },
    },
}


@pytest.fixture()
def input_manifest():
    with open(EXAMPLES_PATH / "input_manifest.json") as f:
        return json.load(f)


def examples(manifest_examples):
    """Returns examples as tuples of component and expected output_manifest."""
    for directory_name, produces_dict in manifest_examples.items():
        directory_path = EXAMPLES_PATH / directory_name
        produces = produces_dict["produces"]
        with open(directory_path / "component.yaml") as c, open(
            directory_path / "output_manifest.json",
        ) as o:
            yield yaml.safe_load(c), json.load(o), produces


@pytest.mark.parametrize(
    ("component_spec", "output_manifest", "produces"),
    list(examples(VALID_EXAMPLE)),
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


@pytest.mark.parametrize(
    ("component_spec", "output_manifest", "produces"),
    list(examples(INVALID_EXAMPLES)),
)
def test_invalid_evolution_examples(
    input_manifest,
    component_spec,
    output_manifest,
    produces,
):
    run_id = "custom_run_id"
    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(component_spec)
    with pytest.raises(InvalidPipelineDefinition):
        manifest.evolve(
            component_spec=component_spec,
            run_id=run_id,
            produces=produces,
        )


def test_component_spec_location_update():
    with open(EXAMPLES_PATH / "input_manifest.json") as f:
        input_manifest = json.load(f)

    with open(EXAMPLES_PATH / "4/component.yaml") as f:
        specification = yaml.safe_load(f)

    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(specification)
    evolved_manifest = manifest.evolve(
        component_spec=component_spec,
    )

    assert evolved_manifest.index.location == "/" + component_spec.component_folder_name
