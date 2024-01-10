# Caption images

<a id="caption_images#description"></a>
## Description
This component captions images using a BLIP model from the Hugging Face hub

<a id="caption_images#inputs_outputs"></a>
## Inputs / outputs 

<a id="caption_images#consumes"></a>
### Consumes 
**This component consumes:**

- image: binary




<a id="caption_images#produces"></a>  
### Produces 
**This component produces:**

- caption: string



<a id="caption_images#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| model_id | str | Id of the BLIP model on the Hugging Face hub | Salesforce/blip-image-captioning-base |
| batch_size | int | Batch size to use for inference | 8 |
| max_new_tokens | int | Maximum token length of each caption | 50 |

<a id="caption_images#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "caption_images",
    arguments={
        # Add arguments
        # "model_id": "Salesforce/blip-image-captioning-base",
        # "batch_size": 8,
        # "max_new_tokens": 50,
    },
)
```

<a id="caption_images#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
