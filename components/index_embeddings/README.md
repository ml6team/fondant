# Index embeddings

### Description
Component that takes embeddings of text snippets and indexes them into a weaviate vector database.

### Inputs / outputs

**This component consumes:**

- text
    - data: string
    - embedding: list<item: float>

**This component produces no data.**

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| weaviate_url | str | The URL of the weaviate instance. | http://localhost:8080 |
| model | str | A name for the model used to generate the embeddings.It is used to name the weaviate class and thus has to follow the naming conventions of weaviate. | MiniLM |
| dataset | str | The name of the data source from which the text snippets that were used to generate the embeddings stem.It is used to name the weaviate class and thus has to follow the naming conventions of weaviate. | / |
| vectorizer | dict | The configuration of the weaviate vectorizer (for vectorization with text2vec-* modules) in the class object corresponding to the model used to generate the embeddings. You can find the available modules in the weaviate documentation: https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules. This can be used for smoother interaction with the weaviate instance after indexing e.g. using langchain similarity search. The module used also has to be enabled in your weaviate instance deployment. Can also be {"vectorizer": "none"} if you do not want to make use of this feature. | / |
| overwrite | bool | Whether to overwrite/ re-create the existing weaviate class and its embeddings. | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


index_embeddings_op = ComponentOp.from_registry(
    name="index_embeddings",
    arguments={
        # Add arguments
        # "weaviate_url": "http://localhost:8080",
        # "model": "MiniLM",
        # "dataset": ,
        # "vectorizer": {},
        # "overwrite": False,
    }
)
pipeline.add_op(index_embeddings_op, dependencies=[...])  #Add previous component as dependency
```

