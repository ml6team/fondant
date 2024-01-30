import tempfile

import dask.dataframe as dd
import pandas as pd

from src.main import WriteToFile


def test_write_to_csv():
    """Test case for write to file component."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entries = 10

        dask_dataframe = dd.DataFrame.from_dict(
            {
                "text": [
                    "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo",
                ]
                * entries,
            },
            npartitions=1,
        )

        component = WriteToFile(
            path=tmpdir,
            format="csv",
        )

        component.write(dask_dataframe)

        df = pd.read_csv(tmpdir + "/export-0.csv")
        assert len(df) == entries


def test_write_to_parquet():
    """Test case for write to file component."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entries = 10

        dask_dataframe = dd.DataFrame.from_dict(
            {
                "text": [
                    "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo",
                ]
                * entries,
            },
            npartitions=1,
        )

        component = WriteToFile(
            path=tmpdir,
            format="parquet",
        )

        component.write(dask_dataframe)

        ddf = dd.read_parquet(tmpdir)
        assert len(ddf) == entries
