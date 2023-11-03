import dask.dataframe as dd
import pandas as pd

from src.main import FilterDataset


def test_transform():
    """Test chunk component method."""
    input_dataframe = dd.from_pandas(
        pd.DataFrame(
            {
                "subset_1_column_1": [1, 6, 3, 8, 5],
                "subset_1_column_2": [
                    "some_value",
                    "another_value",
                    "some_other_value",
                    "some_value",
                    "yet_another_value",
                ],
                "subset_2_column_1": ["value1", "value2", "value3", "value4", "value5"],
                "subset_2_column_2": [
                    "text_with_substring",
                    "some_substring_here",
                    "another_text",
                    "random_text",
                    "text_containing_substring",
                ],
                "subset_2_column_3": [10, 15, 10, 20, 10],
            },
            index=pd.Index(["a", "b", "c", "d", "e"], name="id"),
        ),
        npartitions=1,
    )

    filtering_conditions_1 = [
        ("subset_1_column_1", ">", 5),
        ("subset_1_column_2", "in", ["some_value", "another_value"]),
        ("subset_2_column_1", "not in", ["value1", "value2"]),
    ]

    filtering_conditions_2 = [
        ("subset_2_column_2", "contains", "substring"),
        ("subset_2_column_3", "==", 10),
        ("subset_1_column_1", "!=", 3),
        ("subset_1_column_1", "<=", 4),
    ]

    component_1 = FilterDataset(filtering_conditions_1)
    component_2 = FilterDataset(filtering_conditions_2)

    output_dataframe_1 = component_1.transform(input_dataframe).compute()
    output_dataframe_2 = component_2.transform(input_dataframe).compute()

    expected_output_dataframe_1 = pd.DataFrame(
        {
            "subset_1_column_1": [8],
            "subset_1_column_2": ["some_value"],
            "subset_2_column_1": ["value4"],
            "subset_2_column_2": ["random_text"],
            "subset_2_column_3": [20],
        },
        index=pd.Index(["d"], name="id"),
    )

    expected_output_dataframe_2 = pd.DataFrame(
        {
            "subset_1_column_1": [1],
            "subset_1_column_2": ["some_value"],
            "subset_2_column_1": ["value1"],
            "subset_2_column_2": ["text_with_substring"],
            "subset_2_column_3": [10],
        },
        index=pd.Index(["a"], name="id"),
    )

    pd.testing.assert_frame_equal(
        left=output_dataframe_1,
        right=expected_output_dataframe_1,
        check_dtype=False,
    )

    pd.testing.assert_frame_equal(
        left=output_dataframe_2,
        right=expected_output_dataframe_2,
        check_dtype=False,
    )
