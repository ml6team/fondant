# Image resolution extraction

## Description
Component that extracts image resolution data from the images

## Inputs / outputs

### Consumes
**This component consumes:**

- image: binary





### Produces
**This component produces:**

- image: binary
- image_width: int32
- image_height: int32



## Arguments

This component takes no arguments.

## Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "extract_image_resolution",
    arguments={
        # Add arguments
    },
)
```

