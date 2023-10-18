# MinHash generator

### Description
A component that generates minhashes of text.

### Inputs / outputs

**This component consumes:**

- text
    - data: string

**This component produces:**

- text
    - minhash: list<item: uint64>

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
| shingle_ngram_size | int | Define size of ngram used for the shingle generation | 3 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


minhash_generator_op = ComponentOp.from_registry(
    name="minhash_generator",
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
        # "shingle_ngram_size": 3,
    }
)
pipeline.add_op(minhash_generator_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
