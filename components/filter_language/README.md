# Filter languages

### Description
A component that filters text based on the provided language.

### Inputs / outputs

**This component consumes:**

- text: string

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| language | str | A valid language code or identifier (e.g., "en", "fr", "de"). | en |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


filter_language_op = ComponentOp.from_registry(
    name="filter_language",
    arguments={
        # Add arguments
        # "language": "en",
    }
)
pipeline.add_op(filter_language_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
