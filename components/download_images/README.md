# Download images

### Description
Component that downloads images from a list of URLs.

This component takes in image URLs as input and downloads the images, along with some metadata 
(like their height and width). The images are stored in a new colum as bytes objects. This 
component also resizes the images using the 
[resizer](https://github.com/rom1504/img2dataset/blob/main/img2dataset/resizer.py) function 
from the img2dataset library.


### Inputs/Outputs

**The component comsumes:**
- images
  - url: string

**The component produces:**
- images
  - data: binary
  - width: int32
  - height: int32

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description |
| -------- | ---- | ----------- |
| timeout | int | Maximum time (in seconds) to wait when trying to download an image, |
| retries | int | Number of times to retry downloading an image if it fails. |
| n_connections | int | Number of concurrent connections opened per process. Decrease this number if you are running 
into timeout errors. A lower number of connections can increase the success rate but lower 
the throughput.
 |
| image_size | int | Size of the images after resizing. |
| resize_mode | str | Resize mode to use. One of "no", "keep_ratio", "center_crop", "border". |
| resize_only_if_bigger | bool | If True, resize only if image is bigger than image_size. |
| min_image_size | int | Minimum size of the images. |
| max_aspect_ratio | float | Maximum aspect ratio of the images. |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


download_images_op = ComponentOp.from_registry(
    name="download_images",
    arguments={
        # Add arguments
        "timeout": 10,
        "n_connections": 100,
        "image_size": 256,
        "resize_mode": border,
        "resize_only_if_bigger": False,
        "max_aspect_ratio": inf,
    }
)
pipeline.add_op(Download images_op, dependencies=[...])  #Add previous component as dependency
```

### Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```