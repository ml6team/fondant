# load_from_pdf

<a id="load_from_pdf#description"></a>
## Description
Load pdf data stored locally or remote using langchain loaders.


<a id="load_from_pdf#inputs_outputs"></a>
## Inputs / outputs 

<a id="load_from_pdf#consumes"></a>
### Consumes 


**This component does not consume data.**


<a id="load_from_pdf#produces"></a>  
### Produces 
**This component produces:**

- pdf_path: string
- file_name: string
- text: string



<a id="load_from_pdf#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| pdf_path | str | The path to the a pdf file or a folder containing pdf files to load. Can be a local path or a remote path. If the path is remote, the loader class will be determined by the scheme of the path. | / |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | / |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | / |
| n_partitions | int | Number of partitions of the dask dataframe. If not specified, the number of partitions will be equal to the number of CPU cores. Set to high values if the data is large and the pipelineis running out of memory. | / |

<a id="load_from_pdf#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(
    "load_from_pdf",
    arguments={
        # Add arguments
        # "pdf_path": ,
        # "n_rows_to_load": 0,
        # "index_column": ,
        # "n_partitions": 0,
    },
)
```

<a id="load_from_pdf#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
