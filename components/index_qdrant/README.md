# Index Qdrant

## Description {: #description_index_qdrant}
A Fondant component to load textual data and embeddings into a Qdrant database. NOTE: A Qdrant collection has to be created in advance with the appropriate configurations. https://qdrant.tech/documentation/concepts/collections/

## Inputs / outputs  {: #inputs_outputs_index_qdrant}

### Consumes  {: #consumes_index_qdrant}
**This component consumes:**

- text: string
- embedding: list<item: float>





### Produces {: #produces_index_qdrant}


**This component does not produce data.**

## Arguments {: #arguments_index_qdrant}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| collection_name | str | The name of the Qdrant collection to upsert data into. | / |
| location | str | The location of the Qdrant instance. | / |
| batch_size | int | The batch size to use when uploading points to Qdrant. | 64 |
| parallelism | int | The number of parallel workers to use when uploading points to Qdrant. | 1 |
| url | str | Either host or str of 'Optional[scheme], host, Optional[port], Optional[prefix]'. | / |
| port | int | Port of the REST API interface. | 6333 |
| grpc_port | int | Port of the gRPC interface. | 6334 |
| prefer_grpc | bool | If `true` - use gRPC interface whenever possible in custom methods. | / |
| https | bool | If `true` - use HTTPS(SSL) protocol. | / |
| api_key | str | API key for authentication in Qdrant Cloud. | / |
| prefix | str | If set, add `prefix` to the REST URL path. | / |
| timeout | int | Timeout for API requests. | / |
| host | str | Host name of Qdrant service. If url and host are not set, defaults to 'localhost'. | / |
| path | str | Persistence path for QdrantLocal. Eg. `local_data/qdrant` | / |
| force_disable_check_same_thread | bool | Force disable check_same_thread for QdrantLocal sqlite connection. | / |

## Usage {: #usage_index_qdrant}

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(...)

dataset.write(
    "index_qdrant",
    arguments={
        # Add arguments
        # "collection_name": ,
        # "location": ,
        # "batch_size": 64,
        # "parallelism": 1,
        # "url": ,
        # "port": 6333,
        # "grpc_port": 6334,
        # "prefer_grpc": False,
        # "https": False,
        # "api_key": ,
        # "prefix": ,
        # "timeout": 0,
        # "host": ,
        # "path": ,
        # "force_disable_check_same_thread": False,
    },
)
```

## Testing {: #testing_index_qdrant}

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
