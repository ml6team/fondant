# Filter image resolution

## Description
Component that filters images based on minimum size and max aspect ratio

## Inputs / outputs

### Consumes
**This component consumes:**

- image_width: int32
- image_height: int32



### Produces

_**This component does not produce specific data.**_


## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| min_image_dim | int | Minimum image dimension | / |
| max_aspect_ratio | float | Maximum aspect ratio | / |

## Usage

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

