# Filter languages

## Description
A component that filters text based on the provided language.

## Inputs / outputs

### Consumes
**This component consumes:**

- text: string



### Produces

_**This component does not produce specific data.**_


## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| language | str | A valid language code or identifier (e.g., "en", "fr", "de"). | en |

## Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "filter_language",
    arguments={
        # Add arguments
        # "language": "en",
    },
)
```

## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
