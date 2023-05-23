# download_images

### Description
This component takes in image URLs as input and downloads the images, along with some metadata (like their height and width).
The images are stored in a new colum as bytes objects. This component also resizes the images using the [resizer](https://github.com/rom1504/img2dataset/blob/main/img2dataset/resizer.py) function from the img2dataset library.

If the component is unable to retrieve the image at a URL (for any reason), it will return `None` for that particular URL.

### **Inputs/Outputs**

See [`fondant_component.yaml`](fondant_component.yaml) for a more detailed description on all the input/output parameters. 

