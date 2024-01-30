# resize_images

<a id="resize_images#description"></a>
## Description
Component that resizes images based on given width and height

<a id="resize_images#inputs_outputs"></a>
## Inputs / outputs 

<a id="resize_images#consumes"></a>
### Consumes 
**This component consumes:**

- image: binary




<a id="resize_images#produces"></a>  
### Produces 
**This component produces:**

- image: binary



<a id="resize_images#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| resize_width | int | The width to resize to | / |
| resize_height | int | The height to resize to | / |

<a id="resize_images#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "resize_images",
    arguments={
        # Add arguments
        # "resize_width": 0,
        # "resize_height": 0,
    },
)
```

