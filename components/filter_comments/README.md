# Filter comments

### Description
Component that filters code based on the code to comment ratio

### Inputs/Outputs

**The component comsumes:**
- code
  - content: string

**The component produces:**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| min_comments_ratio | float | The minimum code to comment ratio |
| max_comments_ratio | float | The maximum code to comment ratio |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_comments_op = ComponentOp.from_registry(
    name="filter_comments",
    arguments={
        # Add arguments
        "min_comments_ratio": 0.1,
        "max_comments_ratio": 0.9,
    }
)
pipeline.add_op(Filter comments_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```