# Exttact image resolution

## Description {: #description_exttact_image_resolution}
Component that extracts image resolution data from the images

## Inputs / outputs  {: #inputs_outputs_exttact_image_resolution}

### Consumes  {: #consumes_exttact_image_resolution}
**This component consumes:**

- image: binary





### Produces {: #produces_exttact_image_resolution}
**This component produces:**

- image: binary
- image_width: int32
- image_height: int32



## Arguments {: #arguments_exttact_image_resolution}

This component takes no arguments.

## Usage {: #usage_exttact_image_resolution}

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

