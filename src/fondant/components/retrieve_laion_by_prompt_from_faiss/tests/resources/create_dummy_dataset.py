import dask.dataframe as dd
import pandas as pd


def create_image_index_mapping():
    """Create a mapping between image index and image url."""
    data = {
        "image_id": [1, 2, 3, 4, 5],
        "image_url": ["url1", "url2", "url3", "url4", "url5"],
    }

    # Create Dask DataFrame
    ddf = dd.from_pandas(
        pd.DataFrame(data),
        npartitions=1,
    )  # You can adjust the number of partitions as needed

    # Store as Parquet
    ddf.to_parquet("./dataset.parquet")


create_image_index_mapping()
