# embedding_component

### Description

This component extracts [embeddings](https://rom1504.medium.com/image-embeddings-ed1b194d113e)
from the converted images using a [CLIP model](https://huggingface.co/docs/transformers/model_doc/clip).   

Since image embeddings are good at capturing the features of the image in a compact and useful way, they
will be used in the next steps to retrieve images similar to our seed images. 

The CLIP model is downloaded once inside the component during build time (only the vision encoder is downloaded). The images are then embedded and the embeddings are then added to the manifest as a separate 🤗 dataset. The data manifest is then updated with a reference to the location of the embeddings.

### **Inputs/Outputs**

See [`component.yaml`](component.yaml) for a more detailed description on all the input/output parameters. 

### **Practical considerations**

* There exists many variants for the CLIP model, the current variant that is used is the [`Vit-L/14`](https://huggingface.co/openai/clip-vit-large-patch14) variant. The reason for that is that the embeddings produced from this variant are the same ones that are used for building the indices for the [LAION-5B dataset](https://laion.ai/blog/laion-5b/#:~:text=Pre%2DComputed%20Embeddings).

