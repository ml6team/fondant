# Embed text

<a id="embed_text#description"></a>
## Description
Component that generates embeddings of text passages.

<a id="embed_text#inputs_outputs"></a>
## Inputs / outputs 

<a id="embed_text#consumes"></a>
### Consumes 
**This component consumes:**

- text: string




<a id="embed_text#produces"></a>  
### Produces 
**This component produces:**

- embedding: list<item: float>



<a id="embed_text#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_provider | str | The provider of the model - corresponding to langchain embedding classes. Currently the following providers are supported: aleph_alpha, cohere, huggingface, openai, vertexai. | huggingface |
| model | str | The model to generate embeddings from. Choose an available model name to pass to the model provider's langchain embedding class. | / |
| api_keys | dict | The API keys to use for the model provider that are written to environment variables.Pass only the keys required by the model provider or conveniently pass all keys you will ever need. Pay attention how to name the dictionary keys so that they can be used by the model provider. | / |
| auth_kwargs | dict | Additional keyword arguments required for api initialization/authentication. | / |

<a id="embed_text#usage"></a>
## Usage 

You can apply this component to your dataset using the following code:

```python
from fondant.dataset import Dataset


dataset = Dataset.read(...)

dataset = dataset.apply(
    "embed_text",
    arguments={
        # Add arguments
        # "model_provider": "huggingface",
        # "model": ,
        # "api_keys": {},
        # "auth_kwargs": {},
    },
)
```

<a id="embed_text#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
