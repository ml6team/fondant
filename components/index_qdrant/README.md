# Index Qdrant

### Description
A Fondant component to load textual data and embeddings into a [Qdrant](https://qdrant.tech/) database.

### Inputs / outputs

**This component consumes:**

- text
    - data: string
    - embedding: list<item: float>

**This component produces no data.**

> [!IMPORTANT]  
> A Qdrant collection has to created in advance with appropriate vector configurations. Find out how to [here](https://qdrant.tech/documentation/concepts/collections/).

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| collection_name | str | The name of the Qdrant collection to upsert data into. | / |
| location | str | If `:memory:` - use in-memory Qdrant instance else use it as a url parameter. | None |
| batch_size | int | The batch size to use when uploading points to Qdrant. | 100 |
| parallelism | int | The number of parallel workers to use when uploading points to Qdrant. | None |
| url | str | Either host or str of 'Optional[scheme], host, Optional[port], Optional[prefix]'. Eg. `http://localhost:6333` | None |
| port | int | Port of the REST API interface.| 6333 |
| grpc_port | str | Port of the gRPC interface. | 6334 |
| prefer_grpc | bool | If `true` - use gRPC interface whenever possible in custom methods. | False |
| https | bool | If `true` - use HTTPS(SSL) protocol. | False |
| api_key | str | API key for authentication in Qdrant Cloud. | None |
| prefix | str | If set, add `prefix` to the REST URL path. Example: `service/v1` will result in `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API. | None |
| timeout | int | Timeout for REST and gRPC API requests. | 5 for REST, Unlimited for GRPC |
| host | str | Host name of Qdrant service. If url and host are not set, defaults to 'localhost'. | None |
| path | str | Persistence path for QdrantLocal. Eg. `local_data/qdrant` | None |
| force_disable_check_same_thread | bool | Force disable check_same_thread for QdrantLocal sqlite connection. | False |


### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp

index_qdrant_op = ComponentOp.from_registry(
    name="index_qdrant",
    # Add arguments
    arguments={
        "collection_name": "fondant_loaded_data",
        # "location": "http://localhost:6333",
        # "batch_size": 100,
    }
)
pipeline.add_op(index_qdrant_op, dependencies=[...])
```