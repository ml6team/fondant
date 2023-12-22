# Chunk text

## Description {: #description_chunk_text}
Component that chunks text into smaller segments 

This component takes a body of text and chunks into small chunks. The id of the returned dataset
consists of the id of the original document followed by the chunk index.


## Inputs / outputs  {: #inputs_outputs_chunk_text}

### Consumes  {: #consumes_chunk_text}
**This component consumes:**

- text: string





### Produces {: #produces_chunk_text}
**This component produces:**

- text: string
- original_document_id: string



## Arguments {: #arguments_chunk_text}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| chunk_size | int | Maximum size of chunks to return | / |
| chunk_overlap | int | Overlap in characters between chunks | / |

## Usage {: #usage_chunk_text}

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "chunk_text",
    arguments={
        # Add arguments
        # "chunk_size": 0,
        # "chunk_overlap": 0,
    },
)
```

## Testing {: #testing_chunk_text}

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
