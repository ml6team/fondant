from pathlib import Path

import dask.dataframe as dd
import pytest

from fondant.component_spec import FondantComponentSpec
from fondant.data_io import DaskDataLoader, DaskDataWriter
from fondant.manifest import Manifest

manifest_path = Path(__file__).parent / "example_data/manifest.json"
component_spec_path = Path(__file__).parent / "example_data/components/1.yaml"

NUMBER_OF_TEST_ROWS = 151


@pytest.fixture
def manifest():
    return Manifest.from_file(manifest_path)


@pytest.fixture
def component_spec():
    return FondantComponentSpec.from_file(component_spec_path)


@pytest.fixture
def dataframe(manifest, component_spec):
    data_loader = DaskDataLoader(manifest=manifest)
    return data_loader.load_dataframe(spec=component_spec)


def test_load_index(manifest):
    """Test the loading of just the index."""
    data_loader = DaskDataLoader(manifest=manifest)
    index_df = data_loader._load_index()
    assert len(index_df) == NUMBER_OF_TEST_ROWS
    assert index_df.index.name == "id"


def test_load_subset(manifest):
    """Test the loading of one field of a subset."""
    data_loader = DaskDataLoader(manifest=manifest)
    subset_df = data_loader._load_subset(subset_name="types", fields=["Type 1"])
    assert len(subset_df) == NUMBER_OF_TEST_ROWS
    assert list(subset_df.columns) == ["types_Type 1"]


def test_load_dataframe(manifest, component_spec):
    """Test merging of subsets in a dataframe based on a component_spec."""
    dl = DaskDataLoader(manifest=manifest)
    df = dl.load_dataframe(spec=component_spec)
    assert len(df) == NUMBER_OF_TEST_ROWS
    assert list(df.columns) == [
        "source",
        "properties_Name",
        "properties_HP",
        "types_Type 1",
        "types_Type 2",
    ]


def test_write_index(tmp_path_factory, dataframe, manifest):
    """Test writing out the index."""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        data_writer = DaskDataWriter(manifest=manifest)
        # write out index to temp dir
        data_writer.write_index(df=dataframe)
        # read written data and assert
        df = dd.read_parquet(fn / "index")
        assert len(df) == NUMBER_OF_TEST_ROWS
        assert list(df.columns) == ["source"]
        assert df.index.name == "id"


def test_write_subsets(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out subsets."""
    # Dictionary specifying the expected subsets to write and their column names
    subset_columns_dict = {
        "index": ["source"],
        "properties": ["Name", "HP", "source"],
        "types": ["Type 1", "Type 2", "source"],
    }
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        data_writer = DaskDataWriter(manifest=manifest)
        # write out index to temp dir
        data_writer.write_index(df=dataframe)
        # write out subsets to temp dir
        data_writer.write_subsets(df=dataframe, spec=component_spec)
        # read written data and assert
        for subset, subset_columns in subset_columns_dict.items():
            df = dd.read_parquet(fn / subset)
            assert len(df) == NUMBER_OF_TEST_ROWS
            assert list(df.columns) == subset_columns
            assert df.index.name == "id"


def test_write_subsets_invalid(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out subsets but the dataframe columns are incomplete."""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        # Drop one of the columns required in the output
        dataframe = dataframe.drop(["types_Type 2"], axis=1)
        data_writer = DaskDataWriter(manifest=manifest)
        data_writer.write_index(df=dataframe)
        with pytest.raises(ValueError):
            data_writer.write_subsets(df=dataframe, spec=component_spec)
