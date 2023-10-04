# Filter languages

### Description
A component that filters text based on the provided language.

### Inputs/Outputs

**The component comsumes:**
- text
  - data: string

**The component produces:**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| language | str | A valid language code or identifier (e.g., "en", "fr", "de"). |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


language_filter_op = ComponentOp.from_registry(
    name="language_filter",
    arguments={
        # Add arguments
        "language": en,
    }
)
pipeline.add_op(Filter languages_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```