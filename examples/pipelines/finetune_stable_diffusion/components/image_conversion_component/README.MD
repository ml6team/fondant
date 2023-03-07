# image_conversion_component

### Description

This component converts images from different selected formats
(currently only `png` and `svg`) to `jpg` images. This step is necessary since other images formats 
are not suitable for training ML model on and often contain artifacts that can be eliminated 
during conversion. 

The following image illustrates on the examples of a `png` image (left) when converted to `jgp` during (right) with **normal
conversion**:
```commandline
from PIL import Image

im = Image.open("image.png")
rgb_im = im.convert('RGB')
```

![png_artifacts](docs/png_normal_conversion.png)

You can see that there are some black edge artifacts around the target object. This can also introduce 
subsequent artifacts when inference/finetuning on those images since this is the method of conversion that 
occurs when `png` image are passed as arrays to a model and converted to Tensors. 

For that reason, a [custom function](src/utils/img_conversion.py) is implemented within that component to apply a conversion method
that gets rid of those artifacts. 

### **Inputs/Outputs**
The component accepts as input different parameters, mainly the `file_extensions` parameters which
indicates which image formats to filter. A new `dataset.parquet` file is written with updated reference
to the path of the converted images. 

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters. 

### **Practical considerations**

* `svg` images can be converted and stretched to any desired formats. A good compromise of image quality and file size
is generally `1024x1024` (no risk of distortion if the image is not square sine the conversion just adds a border to compensate)