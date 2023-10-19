# Filter comments

### Description
Component that filters code based on the code to comment ratio

### Inputs / outputs

**This component consumes:**

- code
    - content: string

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| min_comments_ratio | float | The minimum code to comment ratio | 0.1 |
| max_comments_ratio | float | The maximum code to comment ratio | 0.9 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_comments_op = ComponentOp.from_registry(
    name="filter_comments",
    arguments={
        # Add arguments
        # "min_comments_ratio": 0.1,
        # "max_comments_ratio": 0.9,
    }
)
pipeline.add_op(filter_comments_op, dependencies=[...])  #Add previous component as dependency
```

