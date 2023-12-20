# Embed images

## Description
Component that generates CLIP embeddings from images

## Inputs / outputs

### Consumes
**This component consumes:**
- image: binary





### Produces
**This component produces:**
- embedding: list<item: float>



## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | Model id of a CLIP model on the Hugging Face hub | openai/clip-vit-large-patch14 |
| batch_size | int | Batch size to use when embedding | 8 |

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

