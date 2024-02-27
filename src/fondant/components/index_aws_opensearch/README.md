# Index AWS OpenSearch

<a id="index_aws_opensearch#description"></a>
## Description
Component that takes embeddings of text snippets and indexes them into AWS OpenSearch vector database.

<a id="index_aws_opensearch#inputs_outputs"></a>
## Inputs / outputs 

<a id="index_aws_opensearch#consumes"></a>
### Consumes 
**This component consumes:**

- text: string
- embedding: list<item: float>




<a id="index_aws_opensearch#produces"></a>  
### Produces 


**This component does not produce data.**

<a id="index_aws_opensearch#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| host | str | The Cluster endpoint of the AWS OpenSearch cluster where the embeddings will be indexed. E.g. "my-test-domain.us-east-1.aoss.amazonaws.com" | / |
| region | str | The AWS region where the OpenSearch cluster is located. If not specified, the default region will be used. | / |
| index_name | str | The name of the index in the AWS OpenSearch cluster where the embeddings will be stored. | / |
| index_body | dict | Parameters that specify index settings, mappings, and aliases for newly created index. | / |
| port | int | The port number to connect to the AWS OpenSearch cluster. | 443 |
| use_ssl | bool | A boolean flag indicating whether to use SSL/TLS for the connection to the OpenSearch cluster. | True |
| verify_certs | bool | A boolean flag indicating whether to verify SSL certificates when connecting to the OpenSearch cluster. | True |
| pool_maxsize | int | The maximum size of the connection pool to the AWS OpenSearch cluster. | 20 |

<a id="index_aws_opensearch#usage"></a>
## Usage 

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
        # "host": ,
        # "region": ,
        # "index_name": ,
        # "index_body": {},
        # "port": 443,
        # "use_ssl": True,
        # "verify_certs": True,
        # "pool_maxsize": 20,
    },
)
```

<a id="index_aws_opensearch#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
