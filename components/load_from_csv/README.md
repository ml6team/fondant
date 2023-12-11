# Load from csv file

### Description
Component that loads a dataset from a csv file

### Inputs / outputs

**This component consumes no data.**

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| dataset_uri | str | The remote path to the csv file(s) containing the dataset | / |
| column_separator | str | Define the column separator of the csv file | / |
| column_name_mapping | dict | Mapping of the consumed dataset | / |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | / |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(
    "load_from_csv",
    arguments={
        # Add arguments
        # "dataset_uri": ,
        # "column_separator": ,
        # "column_name_mapping": {},
        # "n_rows_to_load": 0,
        # "index_column": ,
    }
)
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
