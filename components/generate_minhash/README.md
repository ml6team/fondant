# Generate minhash

## Description {: #description_generate_minhash}
A component that generates minhashes of text.

## Inputs / outputs  {: #inputs_outputs_generate_minhash}

### Consumes  {: #consumes_generate_minhash}
**This component consumes:**

- text: string





### Produces {: #produces_generate_minhash}
**This component produces:**

- minhash: list<item: uint64>



## Arguments {: #arguments_generate_minhash}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| shingle_ngram_size | int | Define size of ngram used for the shingle generation | 3 |

## Usage {: #usage_generate_minhash}

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "generate_minhash",
    arguments={
        # Add arguments
        # "shingle_ngram_size": 3,
    },
)
```

## Testing {: #testing_generate_minhash}

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
