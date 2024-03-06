# Retrieve images from FAISS index

<a id="retrieve_from_faiss_by_embedding#description"></a>
## Description
Retrieve images from a Faiss index. The component should reference a Faiss image dataset, 
 which includes both the Faiss index and a dataset of image URLs. The input dataset contains 
 embeddings which will be use to retrieve similar images. 


<a id="retrieve_from_faiss_by_embedding#inputs_outputs"></a>
## Inputs / outputs 

<a id="retrieve_from_faiss_by_embedding#consumes"></a>
### Consumes 
**This component consumes:**

- embedding: list<item: float>




<a id="retrieve_from_faiss_by_embedding#produces"></a>  
### Produces 
**This component produces:**

- image_url: string



<a id="retrieve_from_faiss_by_embedding#arguments"></a>
## Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| url_mapping_path | str | Url of the image mapping dataset | / |
| faiss_index_path | str | Url of the dataset | / |
| num_images | int | Number of images that will be retrieved for each prompt | 2 |

<a id="retrieve_from_faiss_by_embedding#usage"></a>
## Usage 

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import Pipeline


pipeline = Pipeline(...)

dataset = pipeline.read(...)

dataset = dataset.apply(
    "retrieve_from_faiss_by_embedding",
    arguments={
        # Add arguments
        # "url_mapping_path": ,
        # "faiss_index_path": ,
        # "num_images": 2,
    },
)
```

<a id="retrieve_from_faiss_by_embedding#testing"></a>
## Testing

You can run the tests using docker with BuildKit. From this directory, run:
```
docker build . --target test
```
