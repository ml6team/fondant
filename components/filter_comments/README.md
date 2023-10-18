# Filter comments

### Description
Component that filters code based on the code to comment ratio

### Inputs / outputs

**This component consumes:**

- code
    - content: string

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
| min_comments_ratio | float | The minimum code to comment ratio | 0.1 |
| max_comments_ratio | float | The maximum code to comment ratio | 0.9 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_comments_op = ComponentOp.from_registry(
    name="filter_comments",
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
        # "min_comments_ratio": 0.1,
        # "max_comments_ratio": 0.9,
    }
)
pipeline.add_op(filter_comments_op, dependencies=[...])  #Add previous component as dependency
```

