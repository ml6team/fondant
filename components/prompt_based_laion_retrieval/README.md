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
| num_images | int | Number of images to retrieve for each prompt | None |
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
        # "num_images": 0,
        # "aesthetic_score": 9,
        # "aesthetic_weight": 0.5,
        # "url": "https://knn.laion.ai/knn-service",
    }
)
pipeline.add_op(prompt_based_laion_retrieval_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```