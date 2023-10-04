# Filter text length

### Description
A component that filters out text based on their length

### Inputs/Outputs

**The component comsumes:**
- text
  - data: string

**The component produces:**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| min_characters_length | int | Minimum number of characters |
| min_words_length | int | Mininum number of words |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


text_length_filter_op = ComponentOp.from_registry(
    name="text_length_filter",
    arguments={
        # Add arguments
    }
)
pipeline.add_op(Filter text length_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```