# retrieve_laion_by_embedding

<a id="retrieve_laion_by_embedding#description"></a>
## Description
This component retrieves image URLs from LAION-5B based on a set of CLIP embeddings. It can be 
used to find images similar to the embedded images / captions.


<a id="retrieve_laion_by_embedding#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_laion_by_embedding#consumes"></a>
### Consumes 
**This component consumes:**

- embedding: list<item: float>




<a id="retrieve_laion_by_embedding#produces"></a>  
### Produces 
**This component produces:**

- image_url: string
- embedding_id: string



<a id="retrieve_laion_by_embedding#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| num_images | int | Number of images to retrieve for each prompt | / |
| aesthetic_score | int | Aesthetic embedding to add to the query embedding, between 0 and 9 (higher is prettier). | 9 |
| aesthetic_weight | float | Weight of the aesthetic embedding when added to the query, between 0 and 1 | 0.5 |

<a id="retrieve_laion_by_embedding#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "retrieve_laion_by_embedding",
    arguments={
        # Add arguments
        # "num_images": 0,
        # "aesthetic_score": 9,
        # "aesthetic_weight": 0.5,
    },
)
```

<a id="retrieve_laion_by_embedding#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
