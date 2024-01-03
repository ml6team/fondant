# Extract image resolution

<a id="extract_image_resolution#description"></a>
## Description
Component that extracts image resolution data from the images

<a id="extract_image_resolution#inputs_outputs"></a>
## Inputs / outputs 

<a id="extract_image_resolution#consumes"></a>
### Consumes 
**This component consumes:**

- image: binary




<a id="extract_image_resolution#produces"></a>  
### Produces 
**This component produces:**

- image: binary
- image_width: int32
- image_height: int32



<a id="extract_image_resolution#arguments"></a>
## Arguments

This component takes no arguments.

<a id="extract_image_resolution#usage"></a>
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

