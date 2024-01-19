# retrieve_from_weaviate

<a id="retrieve_from_weaviate#description"></a>
## Description
Component that retrieves chunks from a Weaviate vector database.
The component can retrieve chunks based on a text search or based on a vector search.
Reranking is only supported for text search.
More info here [Cohere Ranking](https://github.com/weaviate/recipes/blob/main/ranking/cohere-ranking/cohere-ranking.ipynb)
[Weaviate Search Rerank](https://weaviate.io/developers/weaviate/search/rerank)

### Running with text as input

```python
import pyarrow as pa
from fondant.pipeline import Pipeline

pipeline = Pipeline(name="my_pipeline", base_path="path/to/pipeline")

dataset = pipeline.read(
    "load_from_csv",
    arguments={
        "dataset_uri": "path/to/dataset.csv",
    },
    produces={
        "text": pa.string(),
    }
)

dataset = dataset.apply(
    "index_weaviate",
    arguments={
        "weaviate_url": "http://localhost:8080",
        "class_name": "my_class",
        "vectorizer": "text2vec-openai",
        "additional_headers": {
            "X-OpenAI-Api-Key": "YOUR-OPENAI-API-KEY"
        }
    },
    consumes={
        "text": "text"
    }
)

dataset = dataset.apply(
    "retrieve_from_weaviate",
    arguments={
        "weaviate_url": "http://localhost:8080",
        "class_name": "my_class",
        "top_k": 3,
        "additional_headers": {
            "X-OpenAI-Api-Key": "YOUR-OPENAI-API-KEY"
        }
    },
    consumes={
        "text": "text"
    }
)
```

```python
import pyarrow as pa
from fondant.pipeline import Pipeline

pipeline = Pipeline(name="my_pipeline", base_path="path/to/pipeline")

dataset = pipeline.read(
    "load_from_csv",
    arguments={
        "dataset_uri": "path/to/dataset.csv",
    },
    produces={
        "text": pa.string(),
    }
)

dataset = dataset.apply(
    "embed_text",
    arguments={...},
    consumes={
        "text": "text",
    },
)

dataset = dataset.apply(
    "index_weaviate",
    arguments={
        "weaviate_url": "http://localhost:8080",
        "class_name": "my_class",
    },
    consumes={
        "embedding": "embedding"
    }
)

dataset = pipeline.read(
    "load_from_csv",
    arguments={
        "dataset_uri": "path/to/prompt_dataset.csv",
    },
    produces={
        "prompts": pa.string(),
    }
)

dataset = dataset.apply(
    "embed_text",
    arguments={...},
    consumes={
        "prompts": "text",
    },
)

dataset = dataset.apply(
    "retrieve_from_weaviate",
    arguments={
        "weaviate_url": "http://localhost:8080",
        "class_name": "my_class",
        "top_k": 3,
    consumes={
        "text": "text"
    }
)
```


<a id="retrieve_from_weaviate#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_from_weaviate#consumes"></a>
### Consumes 

**This component can consume additional fields**
- <field_name>: <dataset_field_name>
This defines a mapping to update the fields consumed by the operation as defined in the component spec.
The keys are the names of the fields to be received by the component, while the values are 
the name of the field to map from the input dataset

See the usage example below on how to define a field name for additional fields.




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
| additional_config | dict | Additional configuration to pass to the weaviate client. | / |
| additional_headers | dict | Additional headers to pass to the weaviate client. | / |
| hybrid_query | str | The hybrid query to be used for retrieval. Optional parameter. | / |
| hybrid_alpha | float | Argument to change how much each search affects the results. An alpha of 1 is a pure vector search. An alpha of 0 is a pure keyword search. | / |
| rerank | bool | Whether to rerank the results based on the hybrid query. Defaults to False.Check this notebook for more information on reranking:https://github.com/weaviate/recipes/blob/main/ranking/cohere-ranking/cohere-ranking.ipynbhttps://weaviate.io/developers/weaviate/search/rerank. | / |

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
        # "additional_config": {},
        # "additional_headers": {},
        # "hybrid_query": ,
        # "hybrid_alpha": 0.0,
        # "rerank": False,
    },
    consumes={
         <field_name>: <dataset_field_name>,
         ..., # Add fields
     },
)
```

<a id="retrieve_from_weaviate#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
