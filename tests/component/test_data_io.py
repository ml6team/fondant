import os
from pathlib import Path

import dask.dataframe as dd
import pyarrow as pa
import pytest
from dask.distributed import Client
from fondant.component.data_io import DaskDataLoader, DaskDataWriter
from fondant.core.component_spec import ComponentSpec, OperationSpec
from fondant.core.manifest import Manifest

manifest_path = Path(__file__).parent / "examples/data/manifest.json"
component_spec_path = Path(__file__).parent / "examples/data/components/1.yaml"
component_spec_path_custom_consumes = (
    Path(__file__).parent / "examples/data/components/2.yaml"
)
component_spec_path_custom_produces = (
    Path(__file__).parent / "examples/data/components/3.yaml"
)

NUMBER_OF_TEST_ROWS = 151


@pytest.fixture()
def manifest():
    return Manifest.from_file(manifest_path)


@pytest.fixture()
def component_spec():
    return ComponentSpec.from_file(component_spec_path)


@pytest.fixture()
def component_spec_consumes():
    return ComponentSpec.from_file(component_spec_path_custom_consumes)


@pytest.fixture()
def component_spec_produces():
    return ComponentSpec.from_file(component_spec_path_custom_produces)


@pytest.fixture()
def dataframe(manifest, component_spec):
    data_loader = DaskDataLoader(
        manifest=manifest,
        operation_spec=OperationSpec(component_spec),
    )
    return data_loader.load_dataframe()


@pytest.fixture()
def client():
    return Client()


def test_load_dataframe(manifest, component_spec):
    """Test merging of fields in a dataframe based on a component_spec."""
    dl = DaskDataLoader(manifest=manifest, operation_spec=OperationSpec(component_spec))
    dataframe = dl.load_dataframe()
    assert len(dataframe) == NUMBER_OF_TEST_ROWS
    assert list(dataframe.columns) == [
        "Name",
        "HP",
        "Type 1",
        "Type 2",
    ]
    assert dataframe.index.name == "id"


def test_load_dataframe_custom_consumes(manifest, component_spec_consumes):
    """Test loading of columns based on custom defined consumes."""
    consumes = {
        # Consumes remapping (component field <- input dataset field)
        "LastName": "Name",
        "HealthPoints": "HP",
        # Additional columns present in the dataset (defined through generic fields)
        "Type 2": pa.string(),
    }
    dl = DaskDataLoader(
        manifest=manifest,
        operation_spec=OperationSpec(component_spec_consumes, consumes=consumes),
    )
    dataframe = dl.load_dataframe()

    assert list(dataframe.columns) == [
        "LastName",
        "HealthPoints",
        "Type 1",
        "Type 2",
    ]
    assert dataframe.index.name == "id"


def test_load_dataframe_default(manifest, component_spec):
    """Test merging of subsets in a dataframe based on a component_spec."""
    dl = DaskDataLoader(manifest=manifest, operation_spec=OperationSpec(component_spec))
    dataframe = dl.load_dataframe()
    number_workers = os.cpu_count()
    # repartitioning  in dask is an approximation
    assert dataframe.npartitions in list(range(number_workers - 1, number_workers + 2))


def test_load_dataframe_rows(manifest, component_spec):
    """Test merging of fields in a dataframe based on a component_spec."""
    dl = DaskDataLoader(
        manifest=manifest,
        operation_spec=OperationSpec(component_spec),
        input_partition_rows=100,
    )
    dataframe = dl.load_dataframe()
    expected_partitions = 2  # dataset with 151 rows
    assert dataframe.npartitions == expected_partitions


def test_write_dataset(
    tmp_path_factory,
    dataframe,
    manifest,
    component_spec,
    client,
):
    """Test writing out subsets."""
    # Dictionary specifying the expected subsets to write and their column names
    columns = ["Name", "HP", "Type 1", "Type 2"]
    with tmp_path_factory.mktemp("temp") as temp_dir:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(temp_dir))
        data_writer = DaskDataWriter(
            manifest=manifest,
            operation_spec=OperationSpec(component_spec),
        )
        # write dataframe to temp dir
        data_writer.write_dataframe(dataframe)
        # read written data and assert
        dataframe = dd.read_parquet(
            temp_dir
            / manifest.pipeline_name
            / manifest.run_id
            / component_spec.safe_name,
        )
        assert len(dataframe) == NUMBER_OF_TEST_ROWS
        assert list(dataframe.columns) == columns
        assert dataframe.index.name == "id"


def test_write_dataset_custom_produces(
    tmp_path_factory,
    dataframe,
    manifest,
    component_spec_produces,
):
    """Test writing out subsets."""
    produces = {
        # Custom produces (component field -> output dataset field)
        "Name": "LastName",
        "HP": "HealthPoints",
        "Type 1": "CustomFieldName",
        # Additional columns produced in the component and not defined in the dataset
        # (component / output dataset field -> pyarrow data type)
        "Type 2": pa.string(),
    }

    expected_columns = ["LastName", "HealthPoints", "CustomFieldName", "Type 2"]
    with tmp_path_factory.mktemp("temp") as temp_dir:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(temp_dir))
        data_writer = DaskDataWriter(
            manifest=manifest,
            operation_spec=OperationSpec(component_spec_produces, produces=produces),
        )

        # write dataframe to temp dir
        data_writer.write_dataframe(dataframe)
        # # read written data and assert
        dataframe = dd.read_parquet(
            temp_dir
            / manifest.pipeline_name
            / manifest.run_id
            / component_spec_produces.safe_name,
        )
        assert len(dataframe) == NUMBER_OF_TEST_ROWS
        assert list(dataframe.columns) == expected_columns
        assert dataframe.index.name == "id"


# TODO: check if this is still needed?
def test_write_reset_index(
    tmp_path_factory,
    dataframe,
    manifest,
    component_spec,
    client,
):
    """Test writing out the index and fields that have no dask index and checking
    if the id index was created.
    """
    dataframe = dataframe.reset_index(drop=True)
    with tmp_path_factory.mktemp("temp") as fn:
        manifest.update_metadata("base_path", str(fn))

        data_writer = DaskDataWriter(
            manifest=manifest,
            operation_spec=OperationSpec(component_spec),
        )
        data_writer.write_dataframe(dataframe)
        dataframe = dd.read_parquet(fn)
        assert dataframe.index.name == "id"


@pytest.mark.parametrize("partitions", list(range(1, 5)))
def test_write_divisions(  # noqa: PLR0913
    tmp_path_factory,
    dataframe,
    manifest,
    component_spec,
    partitions,
    client,
):
    """Test writing out index and subsets and asserting they have the divisions of the dataframe."""
    # repartition the dataframe (default is 3 partitions)
    dataframe = dataframe.repartition(npartitions=partitions)

    with tmp_path_factory.mktemp("temp") as fn:
        manifest.update_metadata("base_path", str(fn))

        data_writer = DaskDataWriter(
            manifest=manifest,
            operation_spec=OperationSpec(component_spec),
        )

        data_writer.write_dataframe(dataframe)

        dataframe = dd.read_parquet(fn)
        assert dataframe.index.name == "id"
        assert dataframe.npartitions == partitions


def test_write_fields_invalid(
    tmp_path_factory,
    dataframe,
    manifest,
    component_spec,
):
    """Test writing out fields but the dataframe columns are incomplete."""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        # Drop one of the columns required in the output
        dataframe = dataframe.drop(["Type 2"], axis=1)
        data_writer = DaskDataWriter(
            manifest=manifest,
            operation_spec=OperationSpec(component_spec),
        )
        expected_error_msg = (
            r"Fields \['Type 2'\] defined in output dataset "
            r"but not found in dataframe"
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            data_writer.write_dataframe(dataframe)


def test_write_fields_invalid_several_fields_missing(
    tmp_path_factory,
    dataframe,
    manifest,
    component_spec,
):
    """Test writing out fields but the dataframe columns are incomplete."""
    with tmp_path_factory.mktemp("temp") as fn:
        # override the base path of the manifest with the temp dir
        manifest.update_metadata("base_path", str(fn))
        # Drop one of the columns required in the output
        dataframe = dataframe.drop(["Type 1"], axis=1)
        dataframe = dataframe.drop(["Type 2"], axis=1)
        data_writer = DaskDataWriter(
            manifest=manifest,
            operation_spec=OperationSpec(component_spec),
        )
        expected_error_msg = (
            r"Fields \['Type 1', 'Type 2'\] defined in output dataset "
            r"but not found in dataframe"
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            data_writer.write_dataframe(dataframe)
