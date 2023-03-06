# image_embedding_component

### Description

This component extracts [image embeddings](https://rom1504.medium.com/image-embeddings-ed1b194d113e)
from the converted images using a [CLIP model](https://www.google.com/search?q=clip+embeddings&oq=clip+embeddings&aqs=chrome..69i57j0i22i30j69i60j69i64l2j69i60j69i64j69i60.6764j0j7&sourceid=chrome&ie=UTF-8).   

Since image embeddings are good at capturing the features of the image in a compact and useful way, they
will be used in the next steps to retrieve images similar to our seed images. 

The CLIP model is downloaded once inside the component during build time. The images are then embedded
and the embeddings are then saved as numpy files `.npy` files and uploaded to GCS. The data manifest is then updated with the reference to the embeddings path.

### **Inputs/Outputs**

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters. 

### **Practical considerations**

* There exists many variants for the CLIP model, the current variant that is used is the `Vit-L/14`. The reason
for that is that the embeddings produced from this variant are the same ones that are used for building the indices for the 
[LAION-5B dataset](https://laion.ai/blog/laion-5b/#:~:text=Pre%2DComputed%20Embeddings). 

