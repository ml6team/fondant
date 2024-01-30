# load_from_hugging_face_hub

<a id="load_from_hugging_face_hub#description"></a>
## Description
Component that loads a dataset from the hub

<a id="load_from_hugging_face_hub#inputs_outputs"></a>
## Inputs / outputs 

<a id="load_from_hugging_face_hub#consumes"></a>
### Consumes 


**This component does not consume data.**


<a id="load_from_hugging_face_hub#produces"></a>  
### Produces 

**This component can produce additional fields**
- <field_name>: <field_schema>
This defines a mapping to update the fields produced by the operation as defined in the component spec.
The keys are the names of the fields to be produced by the component, while the values are 
the type of the field that should be used to write the output dataset.


<a id="load_from_hugging_face_hub#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| dataset_name | str | Name of dataset on the hub | / |
| column_name_mapping | dict | Mapping of the consumed hub dataset to fondant column names | / |
| image_column_names | list | Optional argument, a list containing the original image column names in case the dataset on the hub contains them. Used to format the image from HF hub format to a byte string. | / |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | / |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | / |

<a id="load_from_hugging_face_hub#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(
    "load_from_hf_hub",
    arguments={
        # Add arguments
        # "dataset_name": ,
        # "column_name_mapping": {},
        # "image_column_names": [],
        # "n_rows_to_load": 0,
        # "index_column": ,
    },
    produces={
         <field_name>: <field_schema>,
         ..., # Add fields
    },
)
```

