# Segment images

## Description
Component that creates segmentation masks for images using a model from the Hugging Face hub

## Inputs / outputs

### Consumes
**This component consumes:**

- image: binary





### Produces
**This component produces:**

- segmentation_map: binary



## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | id of the model on the Hugging Face hub | openmmlab/upernet-convnext-small |
| batch_size | int | batch size to use | 8 |

## Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "segment_images",
    arguments={
        # Add arguments
        # "model_id": "openmmlab/upernet-convnext-small",
        # "batch_size": 8,
    },
)
```

