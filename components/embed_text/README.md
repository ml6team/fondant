# Embed text

### Description
Component that generates embeddings of text passages.

### Inputs / outputs

**This component consumes:**

- text: string

**This component produces:**

- embedding: list<item: float>

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_provider | str | The provider of the model - corresponding to langchain embedding classes. Currently the following providers are supported: aleph_alpha, cohere, huggingface, openai, vertexai. | huggingface |
| model | str | The model to generate embeddings from. Choose an available model name to pass to the model provider's langchain embedding class. | / |
| api_keys | dict | The API keys to use for the model provider that are written to environment variables.Pass only the keys required by the model provider or conveniently pass all keys you will ever need. Pay attention how to name the dictionary keys so that they can be used by the model provider. | / |
| auth_kwargs | dict | Additional keyword arguments required for api initialization/authentication. | / |
| tries | int | Number of retries to attempt when an embedding request fails. | 5 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "embed_text",
    arguments={
        # Add arguments
        # "model_provider": "huggingface",
        # "model": ,
        # "api_keys": {},
        # "auth_kwargs": {},
        # "tries": 5,
    }
)
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
