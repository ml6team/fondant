# Load from parquet

### Description
Component that loads a dataset from a parquet uri

### Inputs / outputs

**This component consumes no data.**

**This component produces:**

- dummy_variable
    - data: binary

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| dataset_uri | str | The remote path to the parquet file/folder containing the dataset | / |
| column_name_mapping | dict | Mapping of the consumed dataset | / |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | / |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


load_from_parquet_op = ComponentOp.from_registry(
    name="load_from_parquet",
    arguments={
        # Add arguments
        # "dataset_uri": ,
        # "column_name_mapping": {},
        # "n_rows_to_load": 0,
        # "index_column": ,
    }
)
pipeline.add_op(load_from_parquet_op, dependencies=[...])  #Add previous component as dependency
```

