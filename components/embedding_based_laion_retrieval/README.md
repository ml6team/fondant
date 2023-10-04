# Embedding based LAION retrieval

### Description
This component retrieves image URLs from LAION-5B based on a set of CLIP embeddings. It can be 
used to find images similar to the embedded images / captions.


### Inputs/Outputs

**The component comsumes:**
- embeddings
  - data: list<item: float>

**The component produces:**
- images
  - url: string

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| num_images | int | Number of images to retrieve for each prompt |
| aesthetic_score | int | Aesthetic embedding to add to the query embedding, between 0 and 9 (higher is prettier). |
| aesthetic_weight | float | Weight of the aesthetic embedding when added to the query, between 0 and 1 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


embedding_based_laion_retrieval_op = ComponentOp.from_registry(
    name="embedding_based_laion_retrieval",
    arguments={
        # Add arguments
        "aesthetic_score": 9,
        "aesthetic_weight": 0.5,
    }
)
pipeline.add_op(Embedding based LAION retrieval_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```