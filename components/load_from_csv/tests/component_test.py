import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa

from src.main import CSVReader


def test_csv_reader():
    input_dataframe = pd.DataFrame.from_dict(
        {
            "id": [0, 1, 2],
            "text": ["zero", "one", "two"],
        },
    )
    input_dataframe = input_dataframe.set_index("id")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        with open(csv_path, "w") as f:
            input_dataframe.to_csv(f)

        component = CSVReader(
            dataset_uri=str(csv_path),
            column_separator=",",
            column_name_mapping={},
            n_rows_to_load=None,
            index_column="id",
            produces={
                "text": pa.string(),
            },
        )

        output_dataframe = component.load().compute()

    pd.testing.assert_frame_equal(
        input_dataframe,
        output_dataframe,
    )
