# PII redaction

### Description
This component detects and redacts Personal Identifiable Information (PII) from code. 
Redaction means that sensitive data is replaced by random data.

The code is based on the PII removal code used as part of the 
[BigCode project](https://github.com/bigcode-project/bigcode-dataset/tree/main/pii).

#### PII detection

The component detects emails, IP addresses and API/SSH keys in text datasets (in particular 
datasets of source code). Regexes are used for emails and IP addresses (they are adapted from 
[BigScience PII pipeline](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/02_pii)). 
The [`detect-secrets`](https://github.com/Yelp/detect-secrets) package is used for finding 
secrets keys. Additionally filters are implemented on top to reduce the number of false 
positives, using the [gibberish-detector](https://github.com/domanchi/gibberish-detector) package.

#### PII redaction

PII is replaced by random data which is stored in the `replacements.json` file.
A component that detects and redacts Personal Identifiable Information (PII) from 
code.


### Inputs / outputs

**This component consumes:**

- code
    - content: string

**This component produces:**

- code
    - content: string

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

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


pii_redaction_op = ComponentOp.from_registry(
    name="pii_redaction",
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
    }
)
pipeline.add_op(pii_redaction_op, dependencies=[...])  #Add previous component as dependency
```

