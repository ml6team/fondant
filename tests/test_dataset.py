import tempfile
import pytest
import dask.dataframe as dd
from pathlib import Path

from fondant.manifest import Manifest
from fondant.dataset import FondantDataset
from fondant.component_spec import FondantComponentSpec

input_manifest_path = Path(__file__).parent / "example_data/input_manifest.json"
output_manifest_path = Path(__file__).parent / "example_data/output_manifest.json"
component_spec_path = Path(__file__).parent / "example_data/components/1.yaml"

DATASET_SIZE = 151


@pytest.fixture
def input_manifest():
    return Manifest.from_file(input_manifest_path)


@pytest.fixture
def output_manifest():
    return Manifest.from_file(input_manifest_path)


@pytest.fixture
def component_spec():
    return FondantComponentSpec.from_file(component_spec_path)


def test_load_index(input_manifest):
    fds = FondantDataset(input_manifest)
    assert len(fds._load_index()) == DATASET_SIZE


def test_merge_subsets(input_manifest, component_spec):
    fds = FondantDataset(manifest=input_manifest)
    df = fds.load_dataframe(spec=component_spec)
    assert len(df) == DATASET_SIZE
    assert list(df.columns) == [
        "id",
        "source",
        "properties_Name",
        "properties_HP",
        "types_Type 1",
        "types_Type 2",
    ]


def test_write_subsets(input_manifest, output_manifest, component_spec):
    # Dictionary specifying the expected subsets to write and their column names
    subset_columns_dict = {
        "index": ["id", "source"],
        "properties": ["Name", "HP", "id", "source"],
        "types": ["Type 1", "Type 2", "id", "source"],
    }

    # Load dataframe from input manifest
    input_fds = FondantDataset(manifest=input_manifest)
    df = input_fds.load_dataframe(spec=component_spec)

    # Write dataframe based on the output manifest and component spec
    output_fds = FondantDataset(manifest=output_manifest)
    output_base_path = Path(output_fds.manifest.base_path)

    # Create temporary directory for writing the subset based on the manifest base path
    with tempfile.TemporaryDirectory(dir=output_base_path):
        tmp_dir_path = Path(output_base_path)
        output_fds.write_index(df)
        output_fds.write_subsets(df, spec=component_spec)
        for subset, subset_columns in subset_columns_dict.items():
            subset_path = Path(tmp_dir_path / subset)
            df = dd.read_parquet(subset_path)
            assert len(df) == DATASET_SIZE
            assert list(df.columns) == subset_columns
