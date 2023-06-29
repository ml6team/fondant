from pathlib import Path

import dask.dataframe as dd
import pytest

from fondant.component_spec import ComponentSpec
from fondant.data_io import DaskDataLoader, DaskDataWriter
from fondant.manifest import Manifest

manifest_path = Path(__file__).parent / "example_data/manifest.json"
component_spec_path = Path(__file__).parent / "example_data/components/1.yaml"

NUMBER_OF_TEST_ROWS = 151


@pytest.fixture()
def manifest():
    return Manifest.from_file(manifest_path)


@pytest.fixture()
def component_spec():
    return ComponentSpec.from_file(component_spec_path)


@pytest.fixture()
def dataframe(manifest, component_spec):
    data_loader = DaskDataLoader(manifest=manifest, component_spec=component_spec)
    return data_loader.load_dataframe()


def test_load_index(manifest, component_spec):
    """Test the loading of just the index."""
    data_loader = DaskDataLoader(manifest=manifest, component_spec=component_spec)
    index_df = data_loader._load_index()
    assert len(index_df) == NUMBER_OF_TEST_ROWS
    assert index_df.index.name == "id"


def test_load_subset(manifest, component_spec):
    """Test the loading of one field of a subset."""
    data_loader = DaskDataLoader(manifest=manifest, component_spec=component_spec)
    subset_df = data_loader._load_subset(subset_name="types", fields=["Type 1"])
    assert len(subset_df) == NUMBER_OF_TEST_ROWS
    assert list(subset_df.columns) == ["types_Type 1"]
    assert subset_df.index.name == "id"


def test_load_dataframe(manifest, component_spec):
    """Test merging of subsets in a dataframe based on a component_spec."""
    dl = DaskDataLoader(manifest=manifest, component_spec=component_spec)
    dataframe = dl.load_dataframe()
    assert len(dataframe) == NUMBER_OF_TEST_ROWS
    assert list(dataframe.columns) == [
        "properties_Name",
        "properties_HP",
        "types_Type 1",
        "types_Type 2",
    ]
    assert dataframe.index.name == "id"


def test_write_index(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out the index."""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        data_writer = DaskDataWriter(manifest=manifest, component_spec=component_spec)
        # write out index to temp dir
        data_writer.write_dataframe(dataframe)
        # read written data and assert
        dataframe = dd.read_parquet(fn / "index")
        assert len(dataframe) == NUMBER_OF_TEST_ROWS
        assert dataframe.index.name == "id"


def test_write_subsets(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out subsets."""
    # Dictionary specifying the expected subsets to write and their column names
    subset_columns_dict = {
        "index": [],
        "properties": ["Name", "HP"],
        "types": ["Type 1", "Type 2"],
    }
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        data_writer = DaskDataWriter(manifest=manifest, component_spec=component_spec)
        # write dataframe to temp dir
        data_writer.write_dataframe(dataframe)
        # read written data and assert
        for subset, subset_columns in subset_columns_dict.items():
            dataframe = dd.read_parquet(fn / subset)
            assert len(dataframe) == NUMBER_OF_TEST_ROWS
            assert list(dataframe.columns) == subset_columns
            assert dataframe.index.name == "id"


def test_write_reset_index(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out the index and subsets that have no dask index and checking
    if the id index was created.
    """
    dataframe = dataframe.reset_index(drop=True)
    with tmp_path_factory.mktemp("temp") as fn:
        manifest.update_metadata("base_path", str(fn))

        data_writer = DaskDataWriter(manifest=manifest, component_spec=component_spec)
        data_writer.write_dataframe(dataframe)

        for subset in ["properties", "types", "index"]:
            dataframe = dd.read_parquet(fn / subset)
            assert dataframe.index.name == "id"


@pytest.mark.parametrize("partitions", list(range(1, 5)))
def test_write_divisions(
    tmp_path_factory,
    dataframe,
    manifest,
    component_spec,
    partitions,
):
    """Test writing out index and subsets and asserting they have the divisions of the dataframe."""
    # repartition the dataframe (default is 3 partitions)
    dataframe = dataframe.repartition(npartitions=partitions)

    with tmp_path_factory.mktemp("temp") as fn:
        manifest.update_metadata("base_path", str(fn))

        data_writer = DaskDataWriter(manifest=manifest, component_spec=component_spec)
        data_writer.write_dataframe(dataframe)

        for target in ["properties", "types", "index"]:
            dataframe = dd.read_parquet(fn / target)
            assert dataframe.index.name == "id"
            assert dataframe.npartitions == partitions


def test_write_subsets_invalid(tmp_path_factory, dataframe, manifest, component_spec):
    """Test writing out subsets but the dataframe columns are incomplete."""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        # Drop one of the columns required in the output
        dataframe = dataframe.drop(["types_Type 2"], axis=1)
        data_writer = DaskDataWriter(manifest=manifest, component_spec=component_spec)
        expected_error_msg = (
            r"Field \['types_Type 2'\] not in index defined in output subset "
            r"types but not found in dataframe"
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            data_writer.write_dataframe(dataframe)
