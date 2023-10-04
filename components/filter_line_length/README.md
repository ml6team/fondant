# Filter line length

### Description
Component that filters code based on line length

### Inputs/Outputs

**The component comsumes:**
- code
  - avg_line_length: double
  - max_line_length: int32
  - alphanum_fraction: double

**The component produces:**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| avg_line_length_threshold | int | Threshold for average line length to filter on |
| max_line_length_threshold | int | Threshold for maximum line length to filter on |
| alphanum_fraction_threshold | float | Alphanum fraction to filter on |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_line_length_op = ComponentOp.from_registry(
    name="filter_line_length",
    arguments={
        # Add arguments
    }
)
pipeline.add_op(Filter line length_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```