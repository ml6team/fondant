# Index Weaviate

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
| batch_size | int | The batch size to be used.Parameter of weaviate.batch.Batch().configure(). | 100 |
| dynamic | bool | Whether to use dynamic batching or not.Parameter of weaviate.batch.Batch().configure(). | True |
| num_workers | int | The maximal number of concurrent threads to run batch import.Parameter of weaviate.batch.Batch().configure(). | 2 |
| overwrite | bool | Whether to overwrite/ re-create the existing weaviate class and its embeddings. | / |
| class_name | str | The name of the weaviate class that will be created and used to store the embeddings.Should follow the weaviate naming conventions. | / |
| vectorizer | str | Which vectorizer to use. You can find the available vectorizers in the weaviate documentation: https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modulesSet this to None if you want to insert your own embeddings. | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


index_weaviate_op = ComponentOp.from_registry(
    name="index_weaviate",
    arguments={
        # Add arguments
        # "weaviate_url": "http://localhost:8080",
        # "batch_size": 100,
        # "dynamic": True,
        # "num_workers": 2,
        # "overwrite": False,
        # "class_name": ,
        # "vectorizer": ,
    }
)
pipeline.add_op(index_weaviate_op, dependencies=[...])  #Add previous component as dependency
```

