# Normalize text

### Description
This component implements several text normalization techniques to clean and preprocess textual 
data:

- Apply lowercasing: Converts all text to lowercase
- Remove unnecessary whitespaces: Eliminates extra spaces between words, e.g. tabs
- Apply NFC normalization: Converts characters to their canonical representation
- Remove common seen patterns in webpages following the implementation of 
  [Penedo et al.](https://arxiv.org/pdf/2306.01116.pdf)
- Remove punctuation: Strips punctuation marks from the text

These text normalization techniques are valuable for preparing text data before using it for
the training of large language models.


### Inputs / outputs

**This component consumes:**

- text
    - data: string

**This component produces no data.**

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
| remove_additional_whitespaces | bool | If true remove all additional whitespace, tabs. | / |
| apply_nfc | bool | If true apply nfc normalization | / |
| normalize_lines | bool | If true analyze documents line-by-line and apply various rules to discard or edit lines. Used to removed common patterns in webpages, e.g. counter | / |
| do_lowercase | bool | If true apply lowercasing | / |
| remove_punctuation | str | If true punctuation will be removed | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


text_normalization_op = ComponentOp.from_registry(
    name="text_normalization",
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
        # "remove_additional_whitespaces": False,
        # "apply_nfc": False,
        # "normalize_lines": False,
        # "do_lowercase": False,
        # "remove_punctuation": ,
    }
)
pipeline.add_op(text_normalization_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
