# Image resolution extraction

### Description
Component that extracts image resolution data from the images

### Inputs / outputs

**This component consumes:**

- images_data: binary

**This component produces:**

- images_data: binary
- images_width: int32
- images_height: int32

### Arguments

This component takes no arguments.

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


image_resolution_extraction_op = ComponentOp.from_registry(
    name="image_resolution_extraction",
    arguments={
        # Add arguments
    }
)
pipeline.add_op(image_resolution_extraction_op, dependencies=[...])  #Add previous component as dependency
```

