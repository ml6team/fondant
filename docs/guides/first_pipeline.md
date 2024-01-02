# Getting started

Fondant provides a couple of [example pipelines](../index.md#example-pipelines) which can help you 
get started.

For demonstration purposes, we'll use the 
[example pipeline](https://github.com/ml6team/fondant-usecase-filter-creative-commons) to download 
and filter images from the 
[fondant-cc-25m](https://huggingface.co/datasets/fondant-ai/fondant-cc-25m) creative commons image 
dataset.

### Clone the Fondant GitHub repository

```
git clone https://github.com/ml6team/fondant-usecase-filter-creative-commons.git
```

### Install the requirements

Install the `requirements.txt`

```
pip install -r requirements.txt
```

And make sure that Docker Compose is [installed](installation.md#docker-installation).

### Run the pipeline


navigate into the `src` folder:
```
cd src
```

And execute the pipeline locally using the fondant cli:
```
fondant run local pipeline.py
```

!!! note "IMPORTANT"   

    For local testing purposes, the pipeline will only download the first 100 images. 

### Inspect the results

Congrats, you just ran your first Fondant pipeline!
To visually inspect the results between every pipeline step, you can use the fondant explorer:
```
fondant explore start --base_path ./data-dir
```

### Building your own pipeline

To learn how to build your own pipeline, you can:
- Check out the 
  [`pipeline.ipynb` notebook](https://github.com/ml6team/fondant-usecase-filter-creative-commons/blob/main/src/notebook.ipynb) 
  in the example repository which runs through the steps to build the pipeline one by one.
- Continue to the next guide on [building your own pipeline](implement_custom_components.md)
