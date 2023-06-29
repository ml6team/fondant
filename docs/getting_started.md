

## Setting up the environment

### Installing fondant

We suggest that you use a [virtual environment](https://docs.python.org/3/library/venv.html) for your project. Fondant supports Python >=3.8.

Fondant can be installed using pip:

```bash
pip install fondant
```

You can validate the installation of fondant by running its root CLI command:
```bash
fondant --help
```

## Your first pipeline

Create a `pipeline.py` file in the root of your project and add the following code:

```Python
from fondant.pipeline import Pipeline, ComponentOp

my_pipeline = Pipeline(
    pipeline_name='my_pipeline',
    base_path='/home/username/my_pipeline',
    pipeline_description='This is my pipeline',
)
```

This is all you need to initialize a fondant pipeline:

- **pipeline_name**: A name to reference your pipeline.
- **base_path**: The base path that fondant should use to store artifacts and data. This base_path can be a local path or a cloud storage path (e.g. s3://my_bucket/artifacts, or gs://my_bucket/artifacts).
- **pipeline_description**: A description of your pipeline.

## Adding components

Now that we have a pipeline, we can add components to it. Components are the building blocks of your pipeline. They are the individual steps that will be executed in your pipeline. There are 2 main types of components:

- reusable components: These are components that are already created by the community and can be easily used in your pipeline. You can find a list of reusable components [here](https://github.com/ml6team/fondant/tree/main/components). They often have arguments that you can set to configure them for your use case.

- custom components: These are the components you create to solve your use case. A custom component can be easily created by adding a `fondant_component.yaml`, `dockerfile` and `main.py` file to your component subdirectory. The `fondant_component.yaml` file contains the specification of your component. You can find more information about it [here](https://github.com/ml6team/fondant/blob/main/docs/component_spec.md)

Let's add a reusable component to our pipeline. We will use the `load_from_hf_hub` component to read data from huggingface. Add the following code to your `pipeline.py` file:

```Python
load_from_hf_hub = ComponentOp.from_registry(
    name='load_from_hf_hub',
    component_spec_path='components/load_from_hf_hub/fondant_component.yml',
    arguments={
        'dataset_name': 'huggan/pokemon',
        'n_rows_to_load': 100,
        'column_name_mapping': {
            'image': 'images_data',
        },
        "image_column_names": ["image"],
        
    }
)

my_pipeline.add_op(load_from_hf_hub, dependencies=[])
```

Two things are happening here:
1. We created a ComponentOp from the registry. This is a reusable component, we pass it arguments needed to configure it

- **dataset_name**: The name of the dataset on huggingface hub, here we load a [dataset with pokemon images](https://huggingface.co/datasets/huggan/pokemon)
- **n_rows_to_load**: The number of rows to load from the dataset. This is useful for testing your pipeline on a small scale.
- **column_name_mapping**: A mapping of the columns in the dataset to the columns in fondant. Here we map the `image` column in the dataset to the `images_raw` subset_column in fondant.
- **image_column_names**: A list of the original image column names in the dataset. This is used to format the image from the huggingface hub format to a byte string.


2. We add our created componentOp to the pipeline using the `add_op` method. This component has no dependencies since it is the first component in our pipeline.


Next create a file `load_from_hf_hub/fondant_component.yml` with the following content:

```yaml
name: Load from hub
description: Component that loads a dataset from the hub
image: ghcr.io/ml6team/load_from_hf_hub:latest

produces:
  images: # subset name
    fields:
      data: # field name
        type: binary # field type


args:
  dataset_name:
    description: Name of dataset on the hub
    type: str
  column_name_mapping:
    description: Mapping of the consumed hub dataset to fondant column names
    type: dict
  image_column_names:
    description: Optional argument, a list containing the original image column names in case the 
      dataset on the hub contains them. Used to format the image from HF hub format to a byte string.
    type: list
    default: None
  n_rows_to_load:
    description: Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale
    type: int
    default: None
```

This is the component spec of the component we have just added to our pipelines, the only thing we have altered is the `produces` section. We have defined what subsets, fields and types this component produces.

Your project should look like this now:
```
.
├── components
│   └── load_from_hf_hub
│       └── fondant_component.yml
└── pipeline.py
```

We now have a fully functional Fondant pipeline, it does not have much functionality yet, but it is a good starting point to build upon. We can already try running this limited example in order to validate our setup.

## Running your pipeline

A Fondant pipeline needs to be compiled before it can be ran. This means translating the user friendly Fondant pipeline definition into something that can be executed by a runner.

There are currently 2 runners available:
- Local runner: This runner runs the pipeline locally on your machine. This is useful for testing your pipeline. We leverage Docker Compose to compile and run the pipeline locally.
- Kubeflow runner: This runner runs the pipeline on a Kubeflow cluster. This is useful for running your pipeline in production on full data.

Fondant has a feature rich CLI that helps you with these steps, let's start by runnin our pipeline with the local runner:

```bash
fondant run pipeline:my_pipeline --local
```

We call the fondant CLI to compile and run our pipeline, we pass a reference to our pipeline using the import_string syntax `<module>:<instance>`. We also pass the `--local` flag to indicate we want to compile our pipeline for the local runner.
Running this command will create a `docker-compose.yml` file with the compiled pipeline definition. Feel free to inspect this file but changing it is not needed.

Note that if you use a local `base_path` in your pipeline declaration that this path will be mounted in the docker containers. This means that the data will be stored locally on your machine. If you use a cloud storage path, the data will be stored in the cloud.

You should see the image used by the component being pulled and a container being created that downloads the dataset from huggingface hub. This container will be removed after the pipeline has finished running. But the data should be stored as parquet files in the `base_path` you defined when creating the pipeline.


## Adding a custom component

Let's expand our pipeline by adding a custom component that will add the height and width of the images as extra columns.


First we create a new folder under `/components` for our new custom component:

```bash
mkdir components/extract_resolution
```

We need to create a couple of things for our custom component:

1. A `fondant_component.yml` file that contains the metadata of our component, this defines the data the component needs (`consumes`) and what data the component produces (`produces`).

```yaml
name: Image resolution extraction
description: Component that extracts image resolution data from the images
image: . 

consumes:
  images:
    fields:
      data:
        type: binary

produces:
  images:
    fields:
      width:
        type: int16
      height:
        type: int16
      data:
        type: binary
```
In our case this component will consume the data field of the images subset and produce the width and height of the images as extra columns.


Now let's create some code (in `extract_resolution/src/main.py`) that will extract the width and height of the images:

```python
"""This component filters images of the dataset based on image size (minimum height and width)."""
import io
import logging
import typing as t

import imagesize
import numpy as np
import pandas as pd

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


def extract_dimensions(images: bytes) -> t.Tuple[np.int16, np.int16]:
    """Extract the width and height of an image.

    Args:
        images: input dataframe with images_data column

    Returns:
        np.int16: width of the image
        np.int16: height of the image
    """
    width, height = imagesize.get(io.BytesIO(images))

    return np.int16(width), np.int16(height)


class ImageResolutionExtractionComponent(PandasTransformComponent):
    """Component that extracts image dimensions."""

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
        Returns:
            dataset.
        """
        logger.info("Filtering dataset...")

        dataframe[[("images", "width"), ("images", "height")]] = \
            dataframe[[("images", "data")]].map(extract_dimensions)

        return dataframe


if __name__ == "__main__":
    component = ImageResolutionExtractionComponent.from_args()
    component.run()

```
This component is rather simple it will take the images as input and extract the width and height of the images. It will then add these columns to the images subset and return the dataframe. We subclass the `PandasTransformComponent` where the user only needs to define the `transform` method. This method will be called with a pandas dataframe as input and should return a pandas dataframe as output. 

The last thing we need for our component is a `Dockerfile` that specifies the steps needed to build the image our component needs:
    
```dockerfile
FROM --platform=linux/amd64 python:3.8-slim

## System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

ENTRYPOINT ["python", "main.py"]
```
There is nothing special about this Dockerfile, it installs the python dependencies and copies over the source code of our component.

And a `requirements.txt` file that specifies the python dependencies of our component:

```
fondant==0.2.0
pyarrow>=7.0
imagesize==1.4.1
```

With our component complete we can now add it to our pipeline definition (`pipeline.py`):

```python
extract_resolution = ComponentOp(
    component_spec_path='components/extract_resolution/fondant_component.yml', 
)

my_pipeline.add_op(load_from_hf_hub) # this line was already there
my_pipeline.add_op(extract_resolution, dependencies=load_from_hf_hub)
```

We add the component to our pipeline definition and specify that it depends on the `load_from_hf_hub` component. This means that the `load_from_hf_hub` component will be executed first and the output of that component will be passed to the `extract_resolution` component.

We can now easily run or new pipeline:

```bash
fondant run pipeline:my_pipeline --local
```

You will see that the components runs sequentially and that each has its own logs. 

Note that with custom components that the image will be built as part of running the pipeline by leveraging a `build` spec in the docker-compose file. This means that you can change the code of your component and run the pipeline again without having to rebuild the image manually.

## Explore the data