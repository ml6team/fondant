import json
import pytest
import dask.dataframe as dd
from pathlib import Path

from fondant.manifest import Manifest
from fondant.dataset import FondantDataset

manifest_path = Path(__file__).parent / "example_data/manifest.json"
dataset_path = Path(__file__).parent / "example_data/151"

@pytest.fixture
def subset_factory(tmp_path_factory):
    fn = tmp_path_factory.mktemp('temp')
    master_df = dd.read_parquet(path=dataset_path / "testset.parquet")

    # create index subset in temporary directory
    index_df = master_df[['id', 'source']]
    index_df.to_parquet(fn / 'index')

    # create subset parquet files in temporary directory
    with open(manifest_path) as o:
        manifest_json = json.load(o)
        for subset_properties in manifest_json['subsets'].values():
            df = master_df[list(subset_properties['fields'].keys())]
            subset_path = fn / subset_properties['location'].replace('/', '')
            df.to_parquet(subset_path)
    return fn

@pytest.fixture
def manifest(subset_factory):
    with open(manifest_path) as f:
        manifest_json = json.load(f)
        # override the base path of the manifest to the temp folder
        manifest_json['metadata']['base_path'] = str(subset_factory)
        return Manifest(manifest_json)

def test_basic(subset_factory):
    df = dd.read_parquet(subset_factory / 'index')
    assert len(df) == 151
    

def test_load_index(manifest):
    fds = FondantDataset(manifest)
    assert len(fds._load_index()) == 151

    