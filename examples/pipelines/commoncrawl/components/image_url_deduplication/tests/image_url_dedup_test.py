from src.main import ImageUrlDeduplication
import dask.dataframe as dd


def test_image_url_deduplication():
    input_data = dd.read_parquet("tests/test_data/part.0.parquet")

    # duplicate whole dataframe
    input_data = dd.concat([input_data, input_data], ignore_index=True)
    input_data_len = len(input_data)
    input_data = input_data.rename(
        columns={"image_url": "image_image_url", "license_type": "image_license_type"}
    )
    component = ImageUrlDeduplication(spec=None)
    ddf = component.transform(input_data)

    # after dedup duplicates should be gone
    assert len(ddf) == input_data_len / 2  # nosec


def test_image_url_deduplication_check_license():
    input_data_by = dd.read_parquet("tests/test_data/part.0.parquet")
    input_data_by["license_type"] = "by"

    input_data_by_nc_nd = dd.read_parquet("tests/test_data/part.0.parquet")
    input_data_by_nc_nd["license_type"] = "by-nc-nd"

    input_data = dd.concat([input_data_by, input_data_by_nc_nd], ignore_index=True)
    input_data = input_data.rename(
        columns={"image_url": "image_image_url", "license_type": "image_license_type"}
    )

    component = ImageUrlDeduplication(spec=None)
    ddf = component.transform(input_data)

    # after dedup duplicates should be gone
    df = ddf.compute()
    assert (df["image_license_type"] == "by-nc-nd").any()  # nosec
