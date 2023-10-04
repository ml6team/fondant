# Caption images

### Description
This component captions images using a BLIP model from the Hugging Face hub

### Inputs/Outputs

**The component comsumes:**
- images
  - data: binary

**The component produces:**
- captions
  - text: string

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| model_id | str | Id of the BLIP model on the Hugging Face hub |
| batch_size | int | Batch size to use for inference |
| max_new_tokens | int | Maximum token length of each caption |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


caption_images_op = ComponentOp.from_registry(
    name="caption_images",
    arguments={
        # Add arguments
        "model_id": Salesforce/blip-image-captioning-base,
        "batch_size": 8,
        "max_new_tokens": 50,
    }
)
pipeline.add_op(Caption images_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```