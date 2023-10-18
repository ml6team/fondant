# Embed images

### Description
Component that generates CLIP embeddings from images

### Inputs / outputs

**This component consumes:**

- images
    - data: binary

**This component produces:**

- embeddings
    - data: list<item: float>

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| input_manifest_path | str | Path to the input manifest | / |
| component_spec | dict | The component specification as a dictionary | / |
| input_partition_rows | int | The number of rows to load per partition.                         Set to override the automatic partitioning | / |
| cache | bool | Set to False to disable caching, True by default. | True |
| cluster_type | str | The cluster type to use for the execution | default |
| client_kwargs | dict | Keyword arguments to pass to the Dask client | / |
| metadata | str | Metadata arguments containing the run id and base path | / |
| output_manifest_path | str | Path to the output manifest | / |
| model_id | str | Model id of a CLIP model on the Hugging Face hub | openai/clip-vit-large-patch14 |
| batch_size | int | Batch size to use when embedding | 8 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


embed_images_op = ComponentOp.from_registry(
    name="embed_images",
    arguments={
        # Add arguments
        # "input_manifest_path": ,
        # "component_spec": {},
        # "input_partition_rows": 0,
        # "cache": True,
        # "cluster_type": "default",
        # "client_kwargs": {},
        # "metadata": ,
        # "output_manifest_path": ,
        # "model_id": "openai/clip-vit-large-patch14",
        # "batch_size": 8,
    }
)
pipeline.add_op(embed_images_op, dependencies=[...])  #Add previous component as dependency
```

