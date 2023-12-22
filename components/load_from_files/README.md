# Load from files

## Description {: #description_load_from_files}
This component loads data from files in a local or remote (AWS S3, Azure Blob storage, GCS) 
location. It supports the following formats: .zip, gzip, tar and tar.gz.


## Inputs / outputs  {: #inputs_outputs_load_from_files}

### Consumes  {: #consumes_load_from_files}


**This component does not consume data.**



### Produces {: #produces_load_from_files}
**This component produces:**

- filename: string
- content: binary



## Arguments {: #arguments_load_from_files}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| directory_uri | str | Local or remote path to the directory containing the files | / |

## Usage {: #usage_load_from_files}

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(
    "load_from_files",
    arguments={
        # Add arguments
        # "directory_uri": ,
    },
)
```

## Testing {: #testing_load_from_files}

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
