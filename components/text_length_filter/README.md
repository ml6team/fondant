# Filter text length

### Description
A component that filters out text based on their length

### Inputs / outputs

**This component consumes:**
- text
  - data: string

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| min_characters_length | int | Minimum number of characters | None |
| min_words_length | int | Mininum number of words | None |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


text_length_filter_op = ComponentOp.from_registry(
    name="text_length_filter",
    arguments={
        # Add arguments
        # "min_characters_length": 0,
        # "min_words_length": 0,
    }
)
pipeline.add_op(text_length_filter_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```