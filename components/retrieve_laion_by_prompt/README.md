# Retrieve LAION by prompt

<a id="retrieve_laion_by_prompt#description"></a>
## Description
This component retrieves image URLs from the [LAION-5B dataset](https://laion.ai/blog/laion-5b/) 
based on text prompts. The retrieval itself is done based on CLIP embeddings similarity between 
the prompt sentences and the captions in the LAION dataset. 

This component doesn’t return the actual images, only URLs.


<a id="retrieve_laion_by_prompt#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_laion_by_prompt#consumes"></a>
### Consumes 
**This component consumes:**

- prompt: string




<a id="retrieve_laion_by_prompt#produces"></a>  
### Produces 
**This component produces:**

- image_url: string
- prompt_id: string



<a id="retrieve_laion_by_prompt#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| num_images | int | Number of images to retrieve for each prompt | / |
| aesthetic_score | int | Aesthetic embedding to add to the query embedding, between 0 and 9 (higher is prettier). | 9 |
| aesthetic_weight | float | Weight of the aesthetic embedding when added to the query, between 0 and 1 | 0.5 |
| url | str | The url of the backend clip retrieval service, defaults to the public service | https://knn.laion.ai/knn-service |

<a id="retrieve_laion_by_prompt#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "retrieve_laion_by_prompt",
    arguments={
        # Add arguments
        # "num_images": 0,
        # "aesthetic_score": 9,
        # "aesthetic_weight": 0.5,
        # "url": "https://knn.laion.ai/knn-service",
    },
)
```

<a id="retrieve_laion_by_prompt#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
