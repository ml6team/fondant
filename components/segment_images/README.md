# Segment images

### Description
Component that creates segmentation masks for images using a model from the Hugging Face hub

### Inputs / outputs

**This component consumes:**
- images
  - data: binary

**This component produces:**
- segmentations
  - data: binary

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | id of the model on the Hugging Face hub | openmmlab/upernet-convnext-small |
| batch_size | int | batch size to use | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


segment_images_op = ComponentOp.from_registry(
    name="segment_images",
    arguments={
        # Add arguments
        # "model_id": "openmmlab/upernet-convnext-small",
        # "batch_size": 0,
    }
)
pipeline.add_op(segment_images_op, dependencies=[...])  #Add previous component as dependency
```

