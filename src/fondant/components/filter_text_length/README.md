# Filter text length

<a id="filter_text_length#description"></a>
## Description
A component that filters out text based on their length

<a id="filter_text_length#inputs_outputs"></a>
## Inputs / outputs 

<a id="filter_text_length#consumes"></a>
### Consumes 
**This component consumes:**

- text: string




<a id="filter_text_length#produces"></a>  
### Produces 


**This component does not produce data.**

<a id="filter_text_length#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| min_characters_length | int | Minimum number of characters | / |
| min_words_length | int | Mininum number of words | / |

<a id="filter_text_length#usage"></a>
## Usage 

You can apply this component to your dataset using the following code:

```python
from fondant.dataset import Dataset


dataset = Dataset.read(...)

dataset = dataset.apply(
    "filter_text_length",
    arguments={
        # Add arguments
        # "min_characters_length": 0,
        # "min_words_length": 0,
    },
)
```

<a id="filter_text_length#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
