# Filter text length

### Description
A component that filters out text based on their length

### Inputs / outputs

**This component consumes:**

- text: string

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| min_characters_length | int | Minimum number of characters | / |
| min_words_length | int | Mininum number of words | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_text_length_op = ComponentOp.from_registry(
    name="filter_text_length",
    arguments={
        # Add arguments
        # "min_characters_length": 0,
        # "min_words_length": 0,
    }
)
pipeline.add_op(filter_text_length_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
