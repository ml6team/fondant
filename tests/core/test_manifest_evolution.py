import json
from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from fondant.core.component_spec import ComponentSpec, OperationSpec
from fondant.core.exceptions import InvalidPipelineDefinition
from fondant.core.manifest import Manifest

EXAMPLES_PATH = Path(__file__).parent / "examples/evolution_examples"

VALID_EXAMPLE = {
    "1": {"produces": None},
    "2": {"produces": None},
    "3": {"produces": None},
    "4": {"produces": None},
    "5": {
        "produces": {
            "embedding_data": pa.list_(pa.float32()),
            "audio_data": "audio_array",
        },
    },
    "6": {
        "produces": {
            "images_data": "images_array",
            "text_data": "text_string",
        },
    },
}

INVALID_EXAMPLES = {
    "1":
    # Invalid data type in the produces mapping
    {
        "produces": {
            "embedding_data": True,
        },
    },
    "3":
    # Non-existent field in the component spec
    {
        "produces": {
            "images_array": "numpy_array",
        },
    },
    "5":
    # Generic produces that has a field in the component mapping that is not in the
    # component spec
    {
        "produces": {
            "images_array": "images_data",
        },
    },
    "6": {
        # Non-generic component that has a type in the produces mapping
        "produces": {
            "embedding_data": pa.list_(pa.float32()),
        },
    },
}


@pytest.fixture()
def input_manifest():
    with open(EXAMPLES_PATH / "input_manifest.json") as f:
        return json.load(f)


def examples(manifest_examples):
    """Returns examples as tuples of component and expected output_manifest."""
    for directory_name, test_conditions in manifest_examples.items():
        directory_path = EXAMPLES_PATH / directory_name
        if isinstance(test_conditions, dict):
            conditions = [test_conditions]
        with open(directory_path / "component.yaml") as c, open(
            directory_path / "output_manifest.json",
        ) as o:
            yield yaml.safe_load(c), json.load(o), conditions


@pytest.mark.parametrize(
    ("component_spec", "output_manifest", "test_conditions"),
    list(examples(VALID_EXAMPLE)),
)
def test_evolution(input_manifest, component_spec, output_manifest, test_conditions):
    run_id = "custom_run_id"
    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(component_spec)
    for test_condition in test_conditions:
        produces = test_condition["produces"]
        operation_spec = OperationSpec(component_spec, produces=produces)
        evolved_manifest = manifest.evolve(
            operation_spec=operation_spec,
            run_id=run_id,
        )
        assert evolved_manifest._specification == output_manifest


@pytest.mark.parametrize(
    ("component_spec", "output_manifest", "test_conditions"),
    list(examples(INVALID_EXAMPLES)),
)
def test_invalid_evolution_examples(
    input_manifest,
    component_spec,
    output_manifest,
    test_conditions,
):
    run_id = "custom_run_id"
    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(component_spec)
    for test_condition in test_conditions:
        produces = test_condition["produces"]
        with pytest.raises(InvalidPipelineDefinition):  # noqa: PT012
            operation_spec = OperationSpec(component_spec, produces=produces)
            manifest.evolve(
                operation_spec=operation_spec,
                run_id=run_id,
            )


def test_component_spec_location_update():
    with open(EXAMPLES_PATH / "input_manifest.json") as f:
        input_manifest = json.load(f)

    with open(EXAMPLES_PATH / "4/component.yaml") as f:
        specification = yaml.safe_load(f)

    manifest = Manifest(input_manifest)
    component_spec = ComponentSpec(specification)
    evolved_manifest = manifest.evolve(
        operation_spec=OperationSpec(component_spec),
        run_id="123",
    )

    assert evolved_manifest.index.location.endswith(
        component_spec.component_folder_name,
    )
