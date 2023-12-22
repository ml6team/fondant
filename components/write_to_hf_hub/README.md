# Write to Hugging Face hub

## Description
Component that writes a dataset to the hub

## Inputs / outputs

### Consumes

**This component can consume additional fields**
- <field_name>: <dataset_field_name>
This defines a mapping to update the fields consumed by the operation as defined in the component spec.
The keys are the names of the fields to be received by the component, while the values are 
the name of the field to map from the input dataset

See the usage example below on how to define a field name for additional fields.





### Produces


**This component does not produce data.**

## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| hf_token | str | The hugging face token used to write to the hub | / |
| username | str | The username under which to upload the dataset | / |
| dataset_name | str | The name of the dataset to upload | / |
| image_column_names | list | A list containing the image column names. Used to format to image to HF hub format | / |
| column_name_mapping | dict | Mapping of the consumed fondant column names to the written hub column names | / |

## Usage

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
    },
    consumes={
         <field_name>: <dataset_field_name>,
         ..., # Add fields
     },
)
```

