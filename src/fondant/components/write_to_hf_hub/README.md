# Write to Hugging Face hub

<a id="write_to_hf_hub#description"></a>
## Description
Component that writes a dataset to the hub

<a id="write_to_hf_hub#inputs_outputs"></a>
## Inputs / outputs 

<a id="write_to_hf_hub#consumes"></a>
### Consumes 

**This component can consume additional fields**
- <field_name>: <dataset_field_name>
This defines a mapping to update the fields consumed by the operation as defined in the component spec.
The keys are the names of the fields to be received by the component, while the values are 
the name of the field to map from the input dataset

See the usage example below on how to define a field name for additional fields.




<a id="write_to_hf_hub#produces"></a>  
### Produces 


**This component does not produce data.**

<a id="write_to_hf_hub#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| hf_token | str | The hugging face token used to write to the hub | / |
| username | str | The username under which to upload the dataset | / |
| dataset_name | str | The name of the dataset to upload | / |
| image_column_names | list | A list containing the image column names. Used to format to image to HF hub format | / |
| column_name_mapping | dict | Mapping of the consumed fondant column names to the written hub column names | / |

<a id="write_to_hf_hub#usage"></a>
## Usage 

You can apply this component to your dataset using the following code:

```python
from fondant.dataset import Dataset


dataset = Dataset.read(...)

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

