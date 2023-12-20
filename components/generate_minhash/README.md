# MinHash generator

## Description
A component that generates minhashes of text.

## Inputs / outputs

### Consumes
**This component consumes:**
- text: string





### Produces
**This component produces:**
- minhash: list<item: uint64>



## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| shingle_ngram_size | int | Define size of ngram used for the shingle generation | 3 |

## Usage

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

## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
