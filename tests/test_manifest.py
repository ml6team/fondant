"""
Test scripts for manifest helpers
"""

import pytest

from express.manifest import DataManifest, DataSource, Metadata, DataType


@pytest.fixture
def valid_manifest_data():
    index = DataSource(location='gs://my-bucket/index.parquet', type=DataType.parquet,
                       extensions=['parquet'])
    data_sources = {
        'source1': DataSource(location='gs://my-bucket/data1.parquet', type=DataType.parquet,
                              extensions=['parquet']),
        'source2': DataSource(location='gs://my-bucket/data2.blob', type=DataType.blob,
                              extensions=['blob'])
    }
    metadata = Metadata(artifact_bucket='gs://my-bucket/artifacts', run_id='12345',
                        component_id='component1',
                        component_name='my-component', branch='main', commit_hash='abc123',
                        creation_date='2022-01-01', num_items=100)
    return {'index': index, 'data_sources': data_sources, 'metadata': metadata}


@pytest.mark.parametrize('invalid_index', [
    DataSource(location='gs://my-bucket/index.csv', type=DataType.blob, extensions=['csv']),
    DataSource(location='gs://my-bucket/index.parquet', type=DataType.blob, extensions=['parquet']),
])
def test_invalid_index(invalid_index, valid_manifest_data):
    valid_manifest_data['index'] = invalid_index
    with pytest.raises(TypeError):
        DataManifest(**valid_manifest_data)


@pytest.mark.parametrize('invalid_data_source_type', [
    DataSource(location='gs://my-bucket/data1.parquet', type='invalid', extensions=['parquet']),
    DataSource(location='gs://my-bucket/data2.blob', type='invalid', extensions=['blob']),
])
def test_invalid_data_source_type(invalid_data_source_type, valid_manifest_data):
    valid_manifest_data['data_sources']['source1'] = invalid_data_source_type
    with pytest.raises(TypeError):
        DataManifest(**valid_manifest_data)


def test_valid_manifest(valid_manifest_data):
    manifest = DataManifest(**valid_manifest_data)
    assert manifest.index == valid_manifest_data['index']
    assert manifest.data_sources == valid_manifest_data['data_sources']
    assert manifest.metadata == valid_manifest_data['metadata']
