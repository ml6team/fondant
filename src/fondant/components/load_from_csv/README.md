# Load from csv

<a id="load_from_csv#description"></a>
## Description
Component that loads a dataset from a csv file

<a id="load_from_csv#inputs_outputs"></a>
## Inputs / outputs 

<a id="load_from_csv#consumes"></a>
### Consumes 


**This component does not consume data.**


<a id="load_from_csv#produces"></a>  
### Produces 

**This component can produce additional fields**
- <field_name>: <field_schema>
This defines a mapping to update the fields produced by the operation as defined in the component spec.
The keys are the names of the fields to be produced by the component, while the values are 
the type of the field that should be used to write the output dataset.


<a id="load_from_csv#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| dataset_uri | str | The remote path to the csv file(s) containing the dataset | / |
| column_separator | str | Define the column separator of the csv file | / |
| column_name_mapping | dict | Mapping of the consumed dataset | / |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing dataset workflows on a small scale | / |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | / |

<a id="load_from_csv#usage"></a>
## Usage 

You can apply this component to your dataset using the following code:

```python
from fondant.dataset import Dataset


dataset = Dataset.create(
    "load_from_csv",
    arguments={
        # Add arguments
        # "dataset_uri": ,
        # "column_separator": ,
        # "column_name_mapping": {},
        # "n_rows_to_load": 0,
        # "index_column": ,
    },
    produces={
         <field_name>: <field_schema>,
         ..., # Add fields
    },
)
```

<a id="load_from_csv#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
