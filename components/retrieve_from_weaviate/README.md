# retrieve_from_weaviate

<a id="retrieve_from_weaviate#description"></a>
## Description
Component that retrieves chunks from a weaviate vectorDB

<a id="retrieve_from_weaviate#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_from_weaviate#consumes"></a>
### Consumes 
**This component consumes:**

- embedding: list<item: float>




<a id="retrieve_from_weaviate#produces"></a>  
### Produces 
**This component produces:**

- retrieved_chunks: list<item: string>



<a id="retrieve_from_weaviate#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| weaviate_url | str | The URL of the weaviate instance. | http://localhost:8080 |
| class_name | str | The name of the weaviate class that will be queried | / |
| top_k | int | Number of chunks to retrieve | / |

<a id="retrieve_from_weaviate#usage"></a>
## Usage 

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

<a id="retrieve_from_weaviate#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
