# Write to hub

### Description
Component that writes a dataset to the hub

### Inputs / outputs

**This component consumes:**

- dummy_variable
    - data: binary

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
| hf_token | str | The hugging face token used to write to the hub | / |
| username | str | The username under which to upload the dataset | / |
| dataset_name | str | The name of the dataset to upload | / |
| image_column_names | list | A list containing the image column names. Used to format to image to HF hub format | / |
| column_name_mapping | dict | Mapping of the consumed fondant column names to the written hub column names | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


write_to_hf_hub_op = ComponentOp.from_registry(
    name="write_to_hf_hub",
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
        # "hf_token": ,
        # "username": ,
        # "dataset_name": ,
        # "image_column_names": [],
        # "column_name_mapping": {},
    }
)
pipeline.add_op(write_to_hf_hub_op, dependencies=[...])  #Add previous component as dependency
```

