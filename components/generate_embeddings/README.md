# Generate embeddings

### Description
Component that generates embeddings of text passages.

### Inputs / outputs

**This component consumes:**

- text
    - data: string

**This component produces:**

- text
    - data: string
    - embedding: list<item: float>

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| input_manifest_path | str | Path to the input manifest | / |
| component_spec | dict | The component specification as a dictionary | / |
| input_partition_rows | int | The number of rows to load per partition.                         Set to override the automatic partitioning | / |
| cache | bool | Set to False to disable caching, True by default. | True |
| cluster_type | str | The cluster type to use for the execution | default |
| client_kwargs | dict | Keyword arguments to pass to the Dask client | / |
| metadata | str | Metadata arguments containing the run id and base path | / |
| output_manifest_path | str | Path to the output manifest | / |
| model_provider | str | The provider of the model - corresponding to langchain embedding classes. Currently the following providers are supported: aleph_alpha, cohere, huggingface, openai. | huggingface |
| model | str | The model to generate embeddings from. Choose an available model name to pass to the model provider's langchain embedding class. | all-MiniLM-L6-v2 |
| api_keys | dict | The API keys to use for the model provider that are written to environment variables.Pass only the keys required by the model provider or conveniently pass all keys you will ever need. Pay attention how to name the dictionary keys so that they can be used by the model provider.You can find all supported/ needed key in the default value. | {'ALEPH_ALPHA_API_KEY': 'api_key', 'COHERE_API_KEY': 'api_key', 'OPENAI_API_KEY': 'api_key'} |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


generate_embeddings_op = ComponentOp.from_registry(
    name="generate_embeddings",
    arguments={
        # Add arguments
        # "input_manifest_path": ,
        # "component_spec": {},
        # "input_partition_rows": 0,
        # "cache": True,
        # "cluster_type": "default",
        # "client_kwargs": {},
        # "metadata": ,
        # "output_manifest_path": ,
        # "model_provider": "huggingface",
        # "model": "all-MiniLM-L6-v2",
        # "api_keys": {'ALEPH_ALPHA_API_KEY': 'api_key', 'COHERE_API_KEY': 'api_key', 'OPENAI_API_KEY': 'api_key'},
    }
)
pipeline.add_op(generate_embeddings_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
