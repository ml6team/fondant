# Filter image resolution

### Description
Component that filters images based on minimum size and max aspect ratio

### Inputs / outputs

**This component consumes:**

- images
    - width: int32
    - height: int32

**This component produces no data.**

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
| min_image_dim | int | Minimum image dimension | / |
| max_aspect_ratio | float | Maximum aspect ratio | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_image_resolution_op = ComponentOp.from_registry(
    name="filter_image_resolution",
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
        # "min_image_dim": 0,
        # "max_aspect_ratio": 0.0,
    }
)
pipeline.add_op(filter_image_resolution_op, dependencies=[...])  #Add previous component as dependency
```

