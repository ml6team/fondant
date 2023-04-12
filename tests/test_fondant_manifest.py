import pytest

from express.components.common import FondantManifest
from express.manifest import DataSource, Metadata


def test_empty_manifest():
    """Test the validity of an empty manifest"""
    manifest = FondantManifest()
    assert manifest.index is None
    assert manifest.data_sources == {}
    assert manifest.metadata is not None


def test_manifest_from_data_sources():
    """Test the validity of a manifest with an index"""
    metadata = {"run_id":100}
    index = DataSource(location='gs://my-bucket/index.parquet', len=100, column_names=['id', 'source'])
    data = DataSource(location='gs://my-bucket/data.parquet', len=100, column_names=['a', 'b', 'c'])
    data_sources = {"dummy": data}

    manifest = FondantManifest(index=index, data_sources=data_sources, metadata=metadata)
    assert manifest.index.location == 'gs://my-bucket/index.parquet'
    assert manifest.data_sources is not None
    assert manifest.metadata.run_id == 100


def test_to_from_json():
    """Test the validity of a manifest from a json file"""
    
    manifest = FondantManifest()
    manifest_json_string = manifest.to_json()
    new_manifest = FondantManifest.from_json(manifest_json_string)