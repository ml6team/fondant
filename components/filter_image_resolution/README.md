# Filter image resolution

## Description {: #description_filter_image_resolution}
Component that filters images based on minimum size and max aspect ratio

## Inputs / outputs  {: #inputs_outputs_filter_image_resolution}

### Consumes  {: #consumes_filter_image_resolution}
**This component consumes:**

- image_width: int32
- image_height: int32





### Produces {: #produces_filter_image_resolution}


**This component does not produce data.**

## Arguments {: #arguments_filter_image_resolution}

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| min_image_dim | int | Minimum image dimension | / |
| max_aspect_ratio | float | Maximum aspect ratio | / |

## Usage {: #usage_filter_image_resolution}

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "filter_image_resolution",
    arguments={
        # Add arguments
        # "min_image_dim": 0,
        # "max_aspect_ratio": 0.0,
    },
)
```

