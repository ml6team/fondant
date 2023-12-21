import dask.dataframe as dd
import numpy as np
import opensearch_py_ml as oml
import pandas as pd

from src.main import IndexAWSOpenSearchComponent


def test_aws_opensearch_write():
    """
    Test case for the write method of the IndexAWSOpenSearchComponent class.

    This test creates an index , inserts dummy data and then deletes the index.
    Writes data to it using the write method of the IndexAWSOpenSearchComponent.
    Asserts that the count of entries in the index is equal to the expected number of entries.
    """
    index_name = "pytest-index"
    aws_os_comp = IndexAWSOpenSearchComponent(
        host="search-genai-vectordb-domain-f7vxqkogveaie2qdrivnkr66om.eu-west-1.es.amazonaws.com",
        port=443,
        region="eu-west-1",
        index_name=index_name,
        index_body={"settings": {"index": {"number_of_shards": 4}}},
        use_ssl=True,
        verify_certs=True,
        pool_maxsize=20,
    )
    pandas_df = pd.DataFrame(
        [
            ("hello abc", np.zeros((1, 1024))),
            ("hifasioi", np.zeros((1, 1024))),
        ],
        columns=["text", "embedding"],
    )
    dask_df = dd.from_pandas(pandas_df, npartitions=2)
    aws_os_comp.write(dask_df)
    oml_df = oml.DataFrame(aws_os_comp.client, aws_os_comp.index_name)
    assert oml_df.shape[0] == pandas_df.shape[0]
    print(
        f"Number of records inserted into index {aws_os_comp.index_name} = {oml_df.shape[0]}",
    )
    aws_os_comp.client.indices.delete(index=index_name)
