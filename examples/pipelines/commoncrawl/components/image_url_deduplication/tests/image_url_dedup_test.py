from src.main import ImageUrlDeduplication
import dask.dataframe as dd


def test_image_url_deduplication():
    input_data = dd.read_parquet("tests/test_data/part.0.parquet")

    # duplicate whole dataframe
    input_data = dd.concat([input_data, input_data], ignore_index=True)
    input_data_len = len(input_data)

    input_data = input_data["image_url"].rename("image_image_url")
    component = ImageUrlDeduplication(spec=None)
    ddf = component.transform(input_data.to_frame())

    # after dedup duplicates should be gone
    assert len(ddf) == input_data_len / 2  # nosec
