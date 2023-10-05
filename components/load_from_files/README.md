# Load from files

### Description
This component loads data from files in a local or remote (AWS S3, Azure Blob storage, GCS) 
location. It supports the following formats: .zip, gzip, tar and tar.gz.


### Inputs / outputs

**This component consumes no data.**

**This component produces:**
- file
  - filename: string
  - content: binary

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| directory_uri | str | Local or remote path to the directory containing the files | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


load_from_files_op = ComponentOp.from_registry(
    name="load_from_files",
    arguments={
        # Add arguments
        # "directory_uri": ,
    }
)
pipeline.add_op(load_from_files_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
