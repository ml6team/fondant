# Getting started

For demonstration purposes, we'll build an 
[example dataset](https://github.com/ml6team/fondant-usecase-filter-creative-commons) with a workflow
that downloads and filter images from the 
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

### Materialize a dataset


navigate into the `src` folder:
```
cd src
```

And materialize the dataset locally using the fondant cli:
```
fondant run local dataset.py --working_directory ./data-dir
```

!!! note "IMPORTANT"   

    For local testing purposes, the workflow will only download the first 100 images. 

### Inspect the results

Congrats, you just materialized your first Fondant dataset!
To visually inspect the results between every workflow step, you can use the fondant explorer:
```
fondant explore start --base_path ./data-dir
```

### Building your own dataset

To learn how to build your own dataset, you can:
- Check out the 
  [`dataset.ipynb` notebook](https://github.com/ml6team/fondant-usecase-filter-creative-commons/blob/main/src/notebook.ipynb) 
  in the example repository which runs through the steps to build the dataset one by one.
- Continue to the next guide on [building your own dataset](implement_custom_components.md)
