# Image cropping

### Description
This component crops out image borders. This is typically useful when working with graphical 
images that have single-color borders (e.g. logos, icons, etc.).

The component takes an image and calculates which color is most present in the border. It then 
crops the image in order to minimize this single-color border. The `padding` argument will add 
extra border to the image before cropping it, in order to avoid cutting off parts of the image.
The resulting crop will always be square. If a crop is not possible, the component will return 
the original image.

#### Examples
Examples of image cropping by removing the single-color border. Left side is original image, 
right side is border-cropped image.

![Example of image cropping by removing the single-color border. Left side is original, right side is cropped image](../../docs/art/components/image_cropping/component_border_crop_1.png)
![Example of image cropping by removing the single-color border. Left side is original, right side is cropped image](../../docs/art/components/image_cropping/component_border_crop_0.png)


### Inputs / outputs

**This component consumes:**
- images
  - data: binary

**This component produces:**
- images
  - data: binary
  - width: int32
  - height: int32

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| cropping_threshold | int | Threshold parameter used for detecting borders. A lower (negative) parameter results in a more performant border detection, but can cause overcropping. Default is -30 | -30 |
| padding | int | Padding for the image cropping. The padding is added to all borders of the image. | 10 |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


image_cropping_op = ComponentOp.from_registry(
    name="image_cropping",
    arguments={
        # Add arguments
        # "cropping_threshold": -30,
        # "padding": 10,
    }
)
pipeline.add_op(image_cropping_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```