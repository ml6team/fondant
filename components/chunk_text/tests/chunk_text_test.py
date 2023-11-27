import pandas as pd

from src.main import ChunkTextComponent


def test_transform():
    """Test chunk component method."""
    input_dataframe = pd.DataFrame(
        {
            "text": [
                "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo",
                "ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis",
                "parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec,",
            ],
        },
        index=pd.Index(["a", "b", "c"], name="id"),
    )

    component = ChunkTextComponent(
        chunk_size=50,
        chunk_overlap=20,
    )

    output_dataframe = component.transform(input_dataframe)

    expected_output_dataframe = pd.DataFrame(
        {
            "original_document_id": ["a", "a", "a", "b", "b", "c", "c"],
            "text": [
                "Lorem ipsum dolor sit amet, consectetuer",
                "amet, consectetuer adipiscing elit. Aenean",
                "elit. Aenean commodo",
                "ligula eget dolor. Aenean massa. Cum sociis",
                "massa. Cum sociis natoque penatibus et magnis dis",
                "parturient montes, nascetur ridiculus mus. Donec",
                "mus. Donec quam felis, ultricies nec,",
            ],
        },
        index=pd.Index(["a_0", "a_1", "a_2", "b_0", "b_1", "c_0", "c_1"], name="id"),
    )

    pd.testing.assert_frame_equal(
        left=output_dataframe,
        right=expected_output_dataframe,
        check_dtype=False,
    )
