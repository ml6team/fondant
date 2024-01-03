# Chunk text

<a id="chunk_text#description"></a>
## Description
Component that chunks text into smaller segments 

This component takes a body of text and chunks into small chunks. The id of the returned dataset
consists of the id of the original document followed by the chunk index.


<a id="chunk_text#inputs_outputs"></a>
## Inputs / outputs 

<a id="chunk_text#consumes"></a>
### Consumes 
**This component consumes:**

- text: string




<a id="chunk_text#produces"></a>  
### Produces 
**This component produces:**

- text: string
- original_document_id: string



<a id="chunk_text#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| chunk_size | int | Maximum size of chunks to return | / |
| chunk_overlap | int | Overlap in characters between chunks | / |

<a id="chunk_text#usage"></a>
## Usage 

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

<a id="chunk_text#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
