# Write to file

<a id="write_to_file#description"></a>
## Description
A Fondant component to write a dataset to file on a local machine or to a cloud storage bucket. The dataset can be written as csv or parquet.

<a id="write_to_file#inputs_outputs"></a>
## Inputs / outputs 

<a id="write_to_file#consumes"></a>
### Consumes 

**This component can consume additional fields**
- <field_name>: <dataset_field_name>
This defines a mapping to update the fields consumed by the operation as defined in the component spec.
The keys are the names of the fields to be received by the component, while the values are 
the name of the field to map from the input dataset

See the usage example below on how to define a field name for additional fields.




<a id="write_to_file#produces"></a>  
### Produces 


**This component does not produce data.**

<a id="write_to_file#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| path | str | Path to store the dataset, whether it's a local path or a cloud storage bucket,  must be specified. A separate filename will be generated for each partition. If you are using the local runner and export the data to a local directory,  ensure that you mount the path to the directory using the `--extra-volumes` argument. | / |
| format | str | Format for storing the dataframe can be either `csv` or `parquet`. As default  `parquet` is used. The CSV files contain the column as a header and use a comma as a delimiter. | parquet |

<a id="write_to_file#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(...)

dataset.write(
    "write_to_file",
    arguments={
        # Add arguments
        # "path": ,
        # "format": "parquet",
    },
    consumes={
         <field_name>: <dataset_field_name>,
         ..., # Add fields
     },
)
```

<a id="write_to_file#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
