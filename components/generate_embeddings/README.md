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
| model | str | The model to generate embeddings from. | all-MiniLM-L6-v2 |
| model_provider | str | The provider of the model. | HuggingFace |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


generate_embeddings_op = ComponentOp.from_registry(
    name="generate_embeddings",
    arguments={
        # Add arguments
        # "model": "all-MiniLM-L6-v2",
        # "model_provider": "HuggingFace",
    }
)
pipeline.add_op(generate_embeddings_op, dependencies=[...])  #Add previous component as dependency
```

