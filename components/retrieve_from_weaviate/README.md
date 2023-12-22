# retrieve_from_weaviate

## Description {: #description_retrieve_from_weaviate}
Component that retrieves chunks from a weaviate vectorDB

## Inputs / outputs  {: #inputs_outputs_retrieve_from_weaviate}

### Consumes  {: #consumes_retrieve_from_weaviate}
**This component consumes:**

- embedding: list<item: float>





### Produces {: #produces_retrieve_from_weaviate}
**This component produces:**

- retrieved_chunks: list<item: string>



## Arguments {: #arguments_retrieve_from_weaviate}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| weaviate_url | str | The URL of the weaviate instance. | http://localhost:8080 |
| class_name | str | The name of the weaviate class that will be queried | / |
| top_k | int | Number of chunks to retrieve | / |

## Usage {: #usage_retrieve_from_weaviate}

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "retrieve_from_weaviate",
    arguments={
        # Add arguments
        # "weaviate_url": "http://localhost:8080",
        # "class_name": ,
        # "top_k": 0,
    },
)
```

## Testing {: #testing_retrieve_from_weaviate}

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
