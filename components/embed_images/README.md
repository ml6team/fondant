# embed_images

<a id="embed_images#description"></a>
## Description
Component that generates CLIP embeddings from images

<a id="embed_images#inputs_outputs"></a>
## Inputs / outputs 

<a id="embed_images#consumes"></a>
### Consumes 
**This component consumes:**

- image: binary




<a id="embed_images#produces"></a>  
### Produces 
**This component produces:**

- embedding: list<item: float>



<a id="embed_images#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | Model id of a CLIP model on the Hugging Face hub | openai/clip-vit-large-patch14 |
| batch_size | int | Batch size to use when embedding | 8 |

<a id="embed_images#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "embed_images",
    arguments={
        # Add arguments
        # "model_id": "openai/clip-vit-large-patch14",
        # "batch_size": 8,
    },
)
```

