# MinHash generator

### Description
A component that generates minhashes of text.

### Inputs / outputs

**This component consumes:**

- text
    - data: string

**This component produces:**

- text
    - minhash: list<item: uint64>

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| shingle_ngram_size | int | Define size of ngram used for the shingle generation | 3 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


minhash_generator_op = ComponentOp.from_registry(
    name="minhash_generator",
    arguments={
        # Add arguments
        # "shingle_ngram_size": 3,
    }
)
pipeline.add_op(minhash_generator_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
