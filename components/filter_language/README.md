# Filter languages

## Description {: #description_filter_languages}
A component that filters text based on the provided language.

## Inputs / outputs  {: #inputs_outputs_filter_languages}

### Consumes  {: #consumes_filter_languages}
**This component consumes:**

- text: string





### Produces {: #produces_filter_languages}


**This component does not produce data.**

## Arguments {: #arguments_filter_languages}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| language | str | A valid language code or identifier (e.g., "en", "fr", "de"). | en |

## Usage {: #usage_filter_languages}

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

## Testing {: #testing_filter_languages}

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
