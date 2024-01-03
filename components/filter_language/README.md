# Filter language

<a id="filter_language#description"></a>
## Description
A component that filters text based on the provided language.

<a id="filter_language#inputs_outputs"></a>
## Inputs / outputs 

<a id="filter_language#consumes"></a>
### Consumes 
**This component consumes:**

- text: string




<a id="filter_language#produces"></a>  
### Produces 


**This component does not produce data.**

<a id="filter_language#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| language | str | A valid language code or identifier (e.g., "en", "fr", "de"). | en |

<a id="filter_language#usage"></a>
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

<a id="filter_language#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
