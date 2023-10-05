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
| dataset_name | str | Name of dataset on the hub | / |
| column_name_mapping | dict | Mapping of the consumed hub dataset to fondant column names | / |
| image_column_names | list | Optional argument, a list containing the original image column names in case the dataset on the hub contains them. Used to format the image from HF hub format to a byte string. | None |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | None |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | None |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


load_from_hf_hub_op = ComponentOp.from_registry(
    name="load_from_hf_hub",
    arguments={
        # Add arguments
        # "dataset_name": ,
        # "column_name_mapping": {},
        # "image_column_names": "None",
        # "n_rows_to_load": "None",
        # "index_column": "None",
    }
)
pipeline.add_op(load_from_hf_hub_op, dependencies=[...])  #Add previous component as dependency
```

