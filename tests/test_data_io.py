from pathlib import Path

import dask.dataframe as dd
import pytest

from fondant.component_spec import FondantComponentSpec
from fondant.data_io import DaskDataLoader, DaskDataWriter
from fondant.manifest import Manifest, Type

manifest_path = Path(__file__).parent / "example_data/manifest.json"
component_spec_path = Path(__file__).parent / "example_data/components/1.yaml"


@pytest.fixture
def manifest():
    return Manifest.from_file(manifest_path)


@pytest.fixture
def component_spec():
    return FondantComponentSpec.from_file(component_spec_path)


@pytest.fixture
def dataframe(manifest, component_spec):
    dl = DaskDataLoader(manifest=manifest)
    return dl.load_dataframe(spec=component_spec)


def test_load_index(manifest):
    """Test the loading of just the index."""
    dl = DaskDataLoader(manifest=manifest)
    assert len(dl._load_index()) == 151


def test_load_subset(manifest):
    """Test the loading of one field of a subset"""
    dl = DaskDataLoader(manifest=manifest)
    subset_df = dl._load_subset(subset_name="types", fields=["Type 1"])
    assert len(subset_df) == 151
    assert list(subset_df.columns) == ["id", "source", "types_Type 1"]


def test_load_dataframe(manifest, component_spec):
    """Test merging of subsets in a dataframe based on a component_spec"""
    dl = DaskDataLoader(manifest=manifest)
    df = dl.load_dataframe(spec=component_spec)
    assert len(df) == 151
    assert list(df.columns) == [
        "id",
        "source",
        "properties_Name",
        "properties_HP",
        "types_Type 1",
        "types_Type 2",
    ]


def test_write_index(tmp_path_factory, dataframe, manifest):
    """Test writing out the index"""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        dw = DaskDataWriter(manifest=manifest)
        # write out index to temp dir
        dw.write_index(df=dataframe)
        # read written data and assert
        odf = dd.read_parquet(fn / "index")
        assert len(odf) == 151
        assert list(odf.columns) == ["id", "source"]


def test_write_subsets(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out subsets."""
    # Dictionary specifying the expected subsets to write and their column names
    subset_columns_dict = {
        "index": ["id", "source"],
        "properties": ["Name", "HP", "id", "source"],
        "types": ["Type 1", "Type 2", "id", "source"],
    }
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        dw = DaskDataWriter(manifest=manifest)
        # write out index to temp dir
        dw.write_index(df=dataframe)
        # write out subsets to temp dir
        dw.write_subsets(df=dataframe, spec=component_spec)
        # read written data and assert
        for subset, subset_columns in subset_columns_dict.items():
            df = dd.read_parquet(fn / subset)
            assert len(df) == 151
            assert list(df.columns) == subset_columns


def test_write_subsets_invalid(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out subsets but the dataframe columns are incomplete."""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        # Drop one of the columns required in the output
        dataframe = dataframe.drop(["types_Type 2"], axis=1)
        dw = DaskDataWriter(manifest=manifest)
        dw.write_index(df=dataframe)
        with pytest.raises(ValueError):
            dw.write_subsets(df=dataframe, spec=component_spec)
