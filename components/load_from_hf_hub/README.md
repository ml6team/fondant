# Load from hub

### Description
Component that loads a dataset from the hub

### Inputs / outputs

**This component consumes no data.**

**This component produces:**

- dummy_variable
    - data: binary

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
| dataset_name | str | Name of dataset on the hub | / |
| column_name_mapping | dict | Mapping of the consumed hub dataset to fondant column names | / |
| image_column_names | list | Optional argument, a list containing the original image column names in case the dataset on the hub contains them. Used to format the image from HF hub format to a byte string. | / |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | / |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


load_from_hf_hub_op = ComponentOp.from_registry(
    name="load_from_hf_hub",
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
        # "dataset_name": ,
        # "column_name_mapping": {},
        # "image_column_names": [],
        # "n_rows_to_load": 0,
        # "index_column": ,
    }
)
pipeline.add_op(load_from_hf_hub_op, dependencies=[...])  #Add previous component as dependency
```

