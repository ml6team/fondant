# Chunk text

### Description
Component that chunks text into smaller segments 

This component takes a body of text and chunks into small chunks. The id of the returned dataset
consists of the id of the original document followed by the chunk index.


### Inputs / outputs

**This component consumes:**

- text: string

**This component produces:**

- text: string
- original_document_id: string

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| chunk_size | int | Maximum size of chunks to return | / |
| chunk_overlap | int | Overlap in characters between chunks | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


chunk_text_op = ComponentOp.from_registry(
    name="chunk_text",
    arguments={
        # Add arguments
        # "chunk_size": 0,
        # "chunk_overlap": 0,
    }
)
pipeline.add_op(chunk_text_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
