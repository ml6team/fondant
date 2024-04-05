# Load from files

<a id="load_from_files#description"></a>
## Description
This component loads data from files in a local or remote (AWS S3, Azure Blob storage, GCS) 
location. It supports the following formats: .zip, gzip, tar and tar.gz.


<a id="load_from_files#inputs_outputs"></a>
## Inputs / outputs 

<a id="load_from_files#consumes"></a>
### Consumes 


**This component does not consume data.**


<a id="load_from_files#produces"></a>  
### Produces 
**This component produces:**

- filename: string
- content: binary



<a id="load_from_files#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| directory_uri | str | Local or remote path to the directory containing the files | / |

<a id="load_from_files#usage"></a>
## Usage 

You can apply this component to your dataset using the following code:

```python
from fondant.dataset import Dataset


dataset = Dataset.create(
    "load_from_files",
    arguments={
        # Add arguments
        # "directory_uri": ,
    },
)
```

<a id="load_from_files#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
