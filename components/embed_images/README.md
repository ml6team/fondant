# Embed images

### Description
Component that generates CLIP embeddings from images

### Inputs / outputs

**This component consumes:**
- images
  - data: binary

**This component produces:**
- embeddings
  - data: list<item: float>

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | Model id of a CLIP model on the Hugging Face hub | openai/clip-vit-large-patch14 |
| batch_size | int | Batch size to use when embedding | 8 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


embed_images_op = ComponentOp.from_registry(
    name="embed_images",
    arguments={
        # Add arguments
        # "model_id": "openai/clip-vit-large-patch14",
        # "batch_size": 8,
    }
)
pipeline.add_op(embed_images_op, dependencies=[...])  #Add previous component as dependency
```

