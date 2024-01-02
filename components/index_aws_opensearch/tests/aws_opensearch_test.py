from unittest.mock import call

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.main import IndexAWSOpenSearchComponent


class TestIndexAWSOpenSearchComponent:
    def setup_method(self):
        self.index_name = "pytest-index"
        self.host = "vectordb-domain-x.eu-west-1.es.amazonaws.com"
        self.region = "eu-west-1"
        self.port = 443
        self.index_body = {"settings": {"index": {"number_of_shards": 4}}}
        self.use_ssl = True
        self.verify_certs = True
        self.pool_maxsize = 20

    def test_create_index(self, mocker):
        # Mock boto3.session
        mocker.patch("src.main.boto3.Session")

        # Mock OpenSearch
        mock_opensearch_instance = mocker.patch("src.main.OpenSearch").return_value
        mock_opensearch_instance.indices.exists.return_value = False

        # Create IndexAWSOpenSearchComponent instance
        IndexAWSOpenSearchComponent(
            host=self.host,
            region=self.region,
            index_name=self.index_name,
            index_body=self.index_body,
            port=self.port,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            pool_maxsize=self.pool_maxsize,
        )

        # Assert that indices.create was called
        mock_opensearch_instance.indices.create.assert_called_once_with(
            index=self.index_name,
            body=self.index_body,
        )

    def test_write(self, mocker):
        # Mock boto3.session
        mocker.patch("src.main.boto3.Session")

        # Mock OpenSearch
        mock_opensearch_instance = mocker.patch("src.main.OpenSearch").return_value
        mock_opensearch_instance.indices.exists.return_value = True

        pandas_df = pd.DataFrame(
            [
                ("hello abc", np.array([1.0, 2.0])),
                ("hifasioi", np.array([2.0, 3.0])),
            ],
            columns=["text", "embedding"],
        )
        dask_df = dd.from_pandas(pandas_df, npartitions=2)

        # Create IndexAWSOpenSearchComponent instance
        component = IndexAWSOpenSearchComponent(
            host=self.host,
            region=self.region,
            index_name=self.index_name,
            index_body=self.index_body,
            port=self.port,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            pool_maxsize=self.pool_maxsize,
        )

        # Call write method
        component.write(dask_df)

        # Assert that index was called with the expected arguments
        expected_calls = [
            call(
                index=self.index_name,
                id="0",
                body={"embedding": np.array([1.0, 2.0]), "text": "hello abc"},
            ),
            call(
                index=self.index_name,
                id="1",
                body={"embedding": np.array([2.0, 3.0]), "text": "hifasioi"},
            ),
        ]

        actual_calls = mock_opensearch_instance.index.call_args_list
        for expected, actual in zip(expected_calls, actual_calls):
            assert expected[2]["index"] == actual[1]["index"]
            assert expected[2]["id"] == actual[1]["id"]
            assert np.array_equal(
                expected[2]["body"]["embedding"],
                actual[1]["body"]["embedding"],
            )
            assert expected[2]["body"]["text"] == actual[1]["body"]["text"]
