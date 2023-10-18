# LAION retrieval

### Description
This component retrieves image URLs from the [LAION-5B dataset](https://laion.ai/blog/laion-5b/) 
based on text prompts. The retrieval itself is done based on CLIP embeddings similarity between 
the prompt sentences and the captions in the LAION dataset. 

This component doesn’t return the actual images, only URLs.


### Inputs / outputs

**This component consumes:**

- prompts
    - text: string

**This component produces:**

- images
    - url: string

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| input_manifest_path | str | Path to the input manifest | / |
| component_spec | dict | The component specification as a dictionary | / |
| input_partition_rows | int | The number of rows to load per partition.                         Set to override the automatic partitioning | / |
| cache | bool | Set to False to disable caching, True by default. | True |
| cluster_type | str | The cluster type to use for the execution | default |
| client_kwargs | dict | Keyword arguments to pass to the Dask client | / |
| metadata | str | Metadata arguments containing the run id and base path | / |
| output_manifest_path | str | Path to the output manifest | / |
| num_images | int | Number of images to retrieve for each prompt | / |
| aesthetic_score | int | Aesthetic embedding to add to the query embedding, between 0 and 9 (higher is prettier). | 9 |
| aesthetic_weight | float | Weight of the aesthetic embedding when added to the query, between 0 and 1 | 0.5 |
| url | str | The url of the backend clip retrieval service, defaults to the public service | https://knn.laion.ai/knn-service |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


prompt_based_laion_retrieval_op = ComponentOp.from_registry(
    name="prompt_based_laion_retrieval",
    arguments={
        # Add arguments
        # "input_manifest_path": ,
        # "component_spec": {},
        # "input_partition_rows": 0,
        # "cache": True,
        # "cluster_type": "default",
        # "client_kwargs": {},
        # "metadata": ,
        # "output_manifest_path": ,
        # "num_images": 0,
        # "aesthetic_score": 9,
        # "aesthetic_weight": 0.5,
        # "url": "https://knn.laion.ai/knn-service",
    }
)
pipeline.add_op(prompt_based_laion_retrieval_op, dependencies=[...])  #Add previous component as dependency
```

