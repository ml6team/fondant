# Retrieve LAION by prompt

<a id="retrieve_laion_by_prompt_from_faiss#description"></a>
## Description
This component retrieves image URLs from the [LAION-5B dataset](https://laion.ai/blog/laion-5b/) 
based on text prompts. The retrieval itself is done based on CLIP embeddings similarity between 
the prompt sentences and the captions in the LAION dataset. 

This component doesn’t return the actual images, only URLs.


<a id="retrieve_laion_by_prompt_from_faiss#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_laion_by_prompt_from_faiss#consumes"></a>
### Consumes 
**This component consumes:**

- prompt: string




<a id="retrieve_laion_by_prompt_from_faiss#produces"></a>  
### Produces 
**This component produces:**

- image_url: string
- prompt_id: string



<a id="retrieve_laion_by_prompt_from_faiss#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| num_images | int | Number of images to retrieve for each prompt | / |
| clip_model | str | Clip model name to use for the retrieval | / |

<a id="retrieve_laion_by_prompt_from_faiss#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "retrieve_laion_by_prompt_from_faiss",
    arguments={
        # Add arguments
        # "num_images": 0,
        # "clip_model": ,
    },
)
```

<a id="retrieve_laion_by_prompt_from_faiss#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
