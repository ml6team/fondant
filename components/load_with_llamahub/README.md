# load_with_llamahub

<a id="load_with_llamahub#description"></a>
## Description
Load data using a LlamaHub loader. For available loaders, check the 
[LlamaHub](https://llamahub.ai/).


<a id="load_with_llamahub#inputs_outputs"></a>
## Inputs / outputs 

<a id="load_with_llamahub#consumes"></a>
### Consumes 


**This component does not consume data.**


<a id="load_with_llamahub#produces"></a>  
### Produces 

**This component can produce additional fields**
- <field_name>: <field_schema>
This defines a mapping to update the fields produced by the operation as defined in the component spec.
The keys are the names of the fields to be produced by the component, while the values are 
the type of the field that should be used to write the output dataset.


<a id="load_with_llamahub#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| loader_class | str | The name of the LlamaIndex loader class to use. Make sure to provide the name and not the id. The name is passed to `llama_index.download_loader` to download the specified loader. | / |
| loader_kwargs | str | Keyword arguments to pass when instantiating the loader class. Check the documentation of the loader to check which arguments it accepts. | / |
| load_kwargs | str | Keyword arguments to pass to the `.load()` method of the loader. Check the documentation ofthe loader to check which arguments it accepts. | / |
| additional_requirements | list | Some loaders require additional dependencies to be installed. You can specify those here. Use a format accepted by `pip install`. Eg. "pypdf" or "pypdf==3.17.1". Unfortunately additional requirements for LlamaIndex loaders are not documented well, but if a dependencyis missing, a clear error message will be thrown. | / |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | / |
| index_column | str | Column to set index to in the load component, if not specified a default globally unique index will be set | / |

<a id="load_with_llamahub#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(
    "load_with_llamahub",
    arguments={
        # Add arguments
        # "loader_class": ,
        # "loader_kwargs": ,
        # "load_kwargs": ,
        # "additional_requirements": [],
        # "n_rows_to_load": 0,
        # "index_column": ,
    },
    produces={
         <field_name>: <field_schema>,
         ..., # Add fields
    },
)
```

<a id="load_with_llamahub#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
