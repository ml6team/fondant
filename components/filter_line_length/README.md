# Filter line length

### Description
Component that filters code based on line length

### Inputs / outputs

**This component consumes:**

- code
    - avg_line_length: double
    - max_line_length: int32
    - alphanum_fraction: double

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
| avg_line_length_threshold | int | Threshold for average line length to filter on | / |
| max_line_length_threshold | int | Threshold for maximum line length to filter on | / |
| alphanum_fraction_threshold | float | Alphanum fraction to filter on | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_line_length_op = ComponentOp.from_registry(
    name="filter_line_length",
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
        # "avg_line_length_threshold": 0,
        # "max_line_length_threshold": 0,
        # "alphanum_fraction_threshold": 0.0,
    }
)
pipeline.add_op(filter_line_length_op, dependencies=[...])  #Add previous component as dependency
```

