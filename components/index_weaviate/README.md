# Index Weaviate

<a id="index_weaviate#description"></a>
## Description
Component that takes text or embeddings of text snippets and indexes them into a Weaviate vector database.

To run the component with text snippets as input, the component needs to be connected to a previous component that outputs text snippets.

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

dataset.write(
    "index_weaviate",
    arguments={
        "weaviate_url": "http://localhost:8080",
        "class_name": "my_class",
        "vectorizer": "text2vec-openai",
        "additional_headers" : {
            "X-OpenAI-Api-Key": "YOUR-OPENAI-API-KEY"
        }
    },
    consumes={
        "text": pa.string()
    }
)
```

### Running with embedding as input

```python
import pyarrow as pa
from fondant.pipeline import Pipeline

pipeline = Pipeline(name="my_pipeline",base_path="path/to/pipeline")

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

dataset.write(
    "index_weaviate",
    arguments={
        "weaviate_url": "http://localhost:8080",
        "class_name": "my_class",
    },
    consumes={
        "embedding": pa.list_(pa.float32())
    }
)
```


<a id="index_weaviate#inputs_outputs"></a>
## Inputs / outputs 

<a id="index_weaviate#consumes"></a>
### Consumes 

**This component can consume additional fields**
- <field_name>: <dataset_field_name>
This defines a mapping to update the fields consumed by the operation as defined in the component spec.
The keys are the names of the fields to be received by the component, while the values are 
the name of the field to map from the input dataset

See the usage example below on how to define a field name for additional fields.




<a id="index_weaviate#produces"></a>  
### Produces 


**This component does not produce data.**

<a id="index_weaviate#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| weaviate_url | str | The URL of the weaviate instance. | http://localhost:8080 |
| batch_size | int | The batch size to be used.Parameter of weaviate.batch.Batch().configure(). | 100 |
| dynamic | bool | Whether to use dynamic batching or not.Parameter of weaviate.batch.Batch().configure(). | True |
| num_workers | int | The maximal number of concurrent threads to run batch import.Parameter of weaviate.batch.Batch().configure(). | 2 |
| overwrite | bool | Whether to overwrite/ re-create the existing weaviate class and its embeddings. | / |
| class_name | str | The name of the weaviate class that will be created and used to store the embeddings.Should follow the weaviate naming conventions. | / |
| additional_config | dict | Additional configuration to pass to the weaviate client. | / |
| additional_headers | dict | Additional headers to pass to the weaviate client. | / |
| vectorizer | str | Which vectorizer to use. You can find the available vectorizers in the weaviate documentation: https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modulesSet this to None if you want to insert your own embeddings. | / |
| module_config | dict | The configuration of the vectorizer module.You can find the available configuration options in the weaviate documentation: https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modulesSet this to None if you want to insert your own embeddings. | / |

<a id="index_weaviate#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(...)

dataset.write(
    "index_weaviate",
    arguments={
        # Add arguments
        # "weaviate_url": "http://localhost:8080",
        # "batch_size": 100,
        # "dynamic": True,
        # "num_workers": 2,
        # "overwrite": False,
        # "class_name": ,
        # "additional_config": {},
        # "additional_headers": {},
        # "vectorizer": ,
        # "module_config": {},
    },
    consumes={
         <field_name>: <dataset_field_name>,
         ..., # Add fields
     },
)
```

<a id="index_weaviate#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
