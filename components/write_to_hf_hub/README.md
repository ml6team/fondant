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
| hf_token | str | The hugging face token used to write to the hub | / |
| username | str | The username under which to upload the dataset | / |
| dataset_name | str | The name of the dataset to upload | / |
| image_column_names | list | A list containing the image column names. Used to format to image to HF hub format | None |
| column_name_mapping | dict | Mapping of the consumed fondant column names to the written hub column names | None |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


write_to_hf_hub_op = ComponentOp.from_registry(
    name="write_to_hf_hub",
    arguments={
        # Add arguments
        # "hf_token": ,
        # "username": ,
        # "dataset_name": ,
        # "image_column_names": "None",
        # "column_name_mapping": "None",
    }
)
pipeline.add_op(write_to_hf_hub_op, dependencies=[...])  #Add previous component as dependency
```

