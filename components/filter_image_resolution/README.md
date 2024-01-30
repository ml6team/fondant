# filter_image_resolution

<a id="filter_image_resolution#description"></a>
## Description
Component that filters images based on minimum size and max aspect ratio

<a id="filter_image_resolution#inputs_outputs"></a>
## Inputs / outputs 

<a id="filter_image_resolution#consumes"></a>
### Consumes 
**This component consumes:**

- image_width: int32
- image_height: int32




<a id="filter_image_resolution#produces"></a>  
### Produces 


**This component does not produce data.**

<a id="filter_image_resolution#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| min_image_dim | int | Minimum image dimension | / |
| max_aspect_ratio | float | Maximum aspect ratio | / |

<a id="filter_image_resolution#usage"></a>
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

