# Index AWS OpenSearch

### Description
Component that takes embeddings of text snippets and indexes them into AWS OpenSearch vector database.

### Inputs / outputs

**This component consumes:**

- text: string
- embedding: list<item: float>

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| host | str | The Cluster endpoint of the AWS OpenSearch cluster where the embeddings will be indexed. For example, "my-opensearch-cluster.us-east-1.es.amazonaws.com" | / |
| region | str | The AWS region where the OpenSearch cluster is located. If not specified, the default region will be used. | / |
| index_name | str | The name of the index in the AWS OpenSearch cluster where the embeddings will be stored. | / |
| index_body | Dict[str, Any] | A dictionary representing the body of the index request. This can include additional settings for the index operation. | / |
| port | int | The port number to connect to the AWS OpenSearch cluster. | 443 |
| use_ssl | bool | A boolean flag indicating whether to use SSL/TLS for the connection to the OpenSearch cluster. | True |
| verify_certs | bool | A boolean flag indicating whether to verify SSL certificates when connecting to the OpenSearch cluster. | True |
| pool_maxsize | int | The maximum size of the connection pool to the AWS OpenSearch cluster. | 20 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline

pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(...)

dataset.write(
    "index_aws_opensearch",
    arguments={
        # Add arguments
        # "host": "my-opensearch-cluster.us-east-1.es.amazonaws.com",
        # "region": "eu-west-1",
        # "index_name": "test-index",
        # "port": 443,
        # "use_ssl": True,
        # "verify_certs": True,
        # "pool_maxsize": 20,
    }
)