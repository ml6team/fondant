# Resize images

### Description
Component that resizes images based on given width and height

### Inputs / outputs

**This component consumes:**

- images
    - data: binary

**This component produces:**

- images
    - data: binary

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| resize_width | int | The width to resize to | / |
| resize_height | int | The height to resize to | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


resize_images_op = ComponentOp.from_registry(
    name="resize_images",
    arguments={
        # Add arguments
        # "resize_width": 0,
        # "resize_height": 0,
    }
)
pipeline.add_op(resize_images_op, dependencies=[...])  #Add previous component as dependency
```

