# Segment images

### Description
Component that creates segmentation masks for images using a model from the Hugging Face hub

### Inputs/Outputs

**The component comsumes:**
- images
  - data: binary

**The component produces:**
- segmentations
  - data: binary

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| model_id | str | id of the model on the Hugging Face hub |
| batch_size | int | batch size to use |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


segment_images_op = ComponentOp.from_registry(
    name="segment_images",
    arguments={
        # Add arguments
        "model_id": openmmlab/upernet-convnext-small,
    }
)
pipeline.add_op(Segment images_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```