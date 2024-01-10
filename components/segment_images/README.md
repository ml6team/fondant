# Segment images

<a id="segment_images#description"></a>
## Description
Component that creates segmentation masks for images using a model from the Hugging Face hub

<a id="segment_images#inputs_outputs"></a>
## Inputs / outputs 

<a id="segment_images#consumes"></a>
### Consumes 
**This component consumes:**

- image: binary




<a id="segment_images#produces"></a>  
### Produces 
**This component produces:**

- segmentation_map: binary



<a id="segment_images#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | id of the model on the Hugging Face hub | openmmlab/upernet-convnext-small |
| batch_size | int | batch size to use | 8 |

<a id="segment_images#usage"></a>
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

