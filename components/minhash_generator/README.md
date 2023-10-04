# MinHash generator

### Description
A component that generates minhashes of text.

### Inputs/Outputs

**The component comsumes:**
- text
  - data: string

**The component produces:**
- text
  - minhash: list<item: uint64>

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| shingle_ngram_size | int | Define size of ngram used for the shingle generation |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


minhash_generator_op = ComponentOp.from_registry(
    name="minhash_generator",
    arguments={
        # Add arguments
        "shingle_ngram_size": 3,
    }
)
pipeline.add_op(MinHash generator_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```