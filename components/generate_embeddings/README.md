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
| model_provider | str | The provider of the model - corresponding to langchain embedding classes. Currently the following providers are supported: aleph_alpha, cohere, huggingface, openai. | huggingface |
| model | str | The model to generate embeddings from. Choose an available model name to pass to the model provider's langchain embedding class. | all-MiniLM-L6-v2 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


generate_embeddings_op = ComponentOp.from_registry(
    name="generate_embeddings",
    arguments={
        # Add arguments
        # "model_provider": "huggingface",
        # "model": "all-MiniLM-L6-v2",
    }
)
pipeline.add_op(generate_embeddings_op, dependencies=[...])  #Add previous component as dependency
```

