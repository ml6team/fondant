# Write to hub

### Description
Component that writes a dataset to the hub

### Inputs / outputs

**This component consumes:**

- dummy_variable: binary

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| hf_token | str | The hugging face token used to write to the hub | / |
| username | str | The username under which to upload the dataset | / |
| dataset_name | str | The name of the dataset to upload | / |
| image_column_names | list | A list containing the image column names. Used to format to image to HF hub format | / |
| column_name_mapping | dict | Mapping of the consumed fondant column names to the written hub column names | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(...)

dataset.write(
    "write_to_hf_hub",
    arguments={
        # Add arguments
        # "hf_token": ,
        # "username": ,
        # "dataset_name": ,
        # "image_column_names": [],
        # "column_name_mapping": {},
    }
)
```

