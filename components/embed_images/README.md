# Embed images

## Description {: #description_embed_images}
Component that generates CLIP embeddings from images

## Inputs / outputs  {: #inputs_outputs_embed_images}

### Consumes  {: #consumes_embed_images}
**This component consumes:**

- image: binary





### Produces {: #produces_embed_images}
**This component produces:**

- embedding: list<item: float>



## Arguments {: #arguments_embed_images}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | Model id of a CLIP model on the Hugging Face hub | openai/clip-vit-large-patch14 |
| batch_size | int | Batch size to use when embedding | 8 |

## Usage {: #usage_embed_images}

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

