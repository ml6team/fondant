# Filter image resolution

### Description
Component that filters images based on minimum size and max aspect ratio

### Inputs/Outputs

**The component comsumes:**
- images
  - width: int32
  - height: int32

**The component produces:**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| min_image_dim | int | Minimum image dimension |
| max_aspect_ratio | float | Maximum aspect ratio |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_image_resolution_op = ComponentOp.from_registry(
    name="filter_image_resolution",
    arguments={
        # Add arguments
    }
)
pipeline.add_op(Filter image resolution_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```