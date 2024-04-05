# Retrieve from FAISS by prompt

<a id="retrieve_from_faiss_by_prompt#description"></a>
## Description
Retrieve images from a Faiss index. The component should reference a Faiss image dataset, 
 which includes both the Faiss index and a dataset of image URLs. The input dataset consists 
 of a list of prompts. These prompts will be embedded using a CLIP model, and similar 
 images will be retrieved from the index.


<a id="retrieve_from_faiss_by_prompt#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_from_faiss_by_prompt#consumes"></a>
### Consumes 
**This component consumes:**

- prompt: string




<a id="retrieve_from_faiss_by_prompt#produces"></a>  
### Produces 
**This component produces:**

- image_url: string
- prompt: string



<a id="retrieve_from_faiss_by_prompt#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| url_mapping_path | str | Url of the image mapping dataset | / |
| faiss_index_path | str | Url of the dataset | / |
| clip_model | str | Clip model name to use for the retrieval | laion/CLIP-ViT-B-32-laion2B-s34B-b79K |
| num_images | int | Number of images that will be retrieved for each prompt | 2 |

<a id="retrieve_from_faiss_by_prompt#usage"></a>
## Usage 

You can apply this component to your dataset using the following code:

```python
from fondant.dataset import Dataset


dataset = Dataset.read(...)

dataset = dataset.apply(
    "retrieve_from_faiss_by_prompt",
    arguments={
        # Add arguments
        # "url_mapping_path": ,
        # "faiss_index_path": ,
        # "clip_model": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        # "num_images": 2,
    },
)
```

<a id="retrieve_from_faiss_by_prompt#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
