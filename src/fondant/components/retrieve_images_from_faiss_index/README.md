# Retrieve images from FAISS index

<a id="retrieve_images_from_faiss_index#description"></a>
## Description
Retrieve images from a Faiss index. The component should reference a Faiss image dataset, 
 which includes both the Faiss index and a dataset of image URLs. The input dataset consists 
 of a list of prompts. These prompts will be embedded using a CLIP model, and similar 
 images will be retrieved from the index.


<a id="retrieve_images_from_faiss_index#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_images_from_faiss_index#consumes"></a>
### Consumes 
**This component consumes:**

- prompt: string




<a id="retrieve_images_from_faiss_index#produces"></a>  
### Produces 
**This component produces:**

- image_url: string
- prompt_id: string



<a id="retrieve_images_from_faiss_index#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| dataset_url | str | Url of the dataset | / |
| faiss_index_path | str | Url of the dataset | / |
| image_index_column_name | str | Name of the column in the dataset that contains the image index | / |
| clip_model | str | Clip model name to use for the retrieval | laion/CLIP-ViT-B-32-laion2B-s34B-b79K |
| num_images | int | Number of images that will be retrieved for each prompt | 2 |

<a id="retrieve_images_from_faiss_index#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "retrieve_images_from_faiss_index",
    arguments={
        # Add arguments
        # "dataset_url": ,
        # "faiss_index_path": ,
        # "image_index_column_name": ,
        # "clip_model": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        # "num_images": 2,
    },
)
```

<a id="retrieve_images_from_faiss_index#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
