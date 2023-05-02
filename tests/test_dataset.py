import json
import pytest
import dask.dataframe as dd
from pathlib import Path

from fondant.manifest import Manifest
from fondant.dataset import FondantDataset
from fondant.component_spec import FondantComponentSpec

manifest_path = Path(__file__).parent / "example_data/manifest.json"
component_spec_path = Path(__file__).parent / "example_data/components/1.yaml"


@pytest.fixture
def manifest():
    return Manifest.from_file(manifest_path)


@pytest.fixture
def component_spec():
    return FondantComponentSpec.from_file(component_spec_path)


def test_load_index(manifest):
    fds = FondantDataset(manifest)
    assert len(fds._load_index()) == 151


def test_merge_subsets(manifest, component_spec):
    fds = FondantDataset(manifest=manifest)
    df = fds.load_dataframe(spec=component_spec)
    assert len(df) == 151
    assert list(df.columns) == [
        "id",
        "source",
        "properties_Name",
        "properties_HP",
        "types_Type 1",
        "types_Type 2",
    ]
