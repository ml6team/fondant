# Resize images

## Description {: #description_resize_images}
Component that resizes images based on given width and height

## Inputs / outputs  {: #inputs_outputs_resize_images}

### Consumes  {: #consumes_resize_images}
**This component consumes:**

- image: binary





### Produces {: #produces_resize_images}
**This component produces:**

- image: binary



## Arguments {: #arguments_resize_images}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| resize_width | int | The width to resize to | / |
| resize_height | int | The height to resize to | / |

## Usage {: #usage_resize_images}

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

