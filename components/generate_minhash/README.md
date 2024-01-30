# Generate minhash

<a id="generate_minhash#description"></a>
## Description
A component that generates minhashes of text.

<a id="generate_minhash#inputs_outputs"></a>
## Inputs / outputs 

<a id="generate_minhash#consumes"></a>
### Consumes 
**This component consumes:**

- text: string




<a id="generate_minhash#produces"></a>  
### Produces 
**This component produces:**

- minhash: list<item: uint64>



<a id="generate_minhash#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| shingle_ngram_size | int | Define size of ngram used for the shingle generation | 3 |

<a id="generate_minhash#usage"></a>
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

<a id="generate_minhash#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
