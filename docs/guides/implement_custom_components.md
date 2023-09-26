# Guide - Implement custom components

**Level**: Beginner </br>
**Time**: 20min </br>
**Goal**: After completing this tutorial with Fondant, you will be able to build your own custom component and integrate it into a fondant pipeline. </br> 

**Prerequisite**: Make sure docker compose is installed on your local system.
We recommend completing the [first tutorial](/docs/guides/build_a_simple_pipeline.md) before proceeding with this one, as this tutorial builds upon the knowledge gained in the previous one.

## Overview

In the [initial tutorial](/docs/guides/build_a_simple_pipeline.md), you learned how to create your first Fondant pipeline. While the example demonstrates initialising the dataset from HuggingFace and using a reusable component to download images, this is just the beginning.

The true power of Fondant lies in its capability to enable you to construct your own data pipelines to create high-quality datasets. To achieve this, we need to implement custom components.

In this tutorial, we will guide you through the process of implementing your very own custom component. We will illustrate this by demonstrating how to filter images based on file type.

This pipeline is an extension of the one introduced in the first tutorial. After loading the dataset from HuggingFace, it narrows down the dataset to exclusively encompass PNG files. Finally, it proceeds to download these carefully selected images.


## Build your custom component

A component comprises several key elements. First, there's the ComponentSpec YAML file, serving as a blueprint for the component. It defines crucial aspects such as input and output dataframes, along with component arguments.

![component architecture](https://github.com/ml6team/fondant/blob/main/docs/art/guides/component.png?raw=true)

The second essential part is a python class, which encapsulates the business logic that operates on the input dataframe.

In addition to these core components, there are a few other necessary items, including a `Dockerfile` used for building the component and a `requirements.txt` file to specify and install required dependencies. You can find a more detailed explanation [here](custom_components.md).

### Creating the ComponentSpec

First of all we create the following ComponentSpec (fondant_component.yaml) file in the folder `components/filter_images`: 

```yaml
name: Filter file type
description: Component that filters on mime types
image: filter_images

consumes:
  images:
    fields:
      url:
        type: string

produces:
  images:
    fields:
      url:
        type: string

args:
  mime_type:
    description: Name of file type
    type: str
```

It begins by specifying the component name, providing a brief description, and naming the component's Docker image url.

Following this, we define the structure of input and output dataframes, consumes and `produces`, which dictate the columns and subset fields the component will operate on. In this example, our goal is to filter images based on file types. For the sake of simplicity, we will work with image URLs, assuming that the file type is identifiable within the URL (e.g., *.png). Consequently, our component consumes image_urls and produces image_urls as well.

Lastly, we define custom arguments that the component will support. In our case, we include the `mime_type argument`, which allows us to filter images by different file types in the future.

### Creating the Component

Now, it's time to outline the component logic. To do this, we'll create a `main.py` file within the `components/filter_images/src directory`:


```python
"""A component that filters images based on file type."""
import logging
import mimetypes
import pandas as pd
from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class FileTypeFilter(PandasTransformComponent):
    """Custom component to filter on specific file type based on url"""

    def __init__(self, *args, mime_type: str):
        self.mime_type = mime_type

    @staticmethod
    def get_mime_type(data):
        """Guess mime type based on the file name"""
        mime_type, _ = mimetypes.guess_type(data)
        return mime_type

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Reduce dataframe to specific mime type"""
        dataframe[("images", "mime_type")] = dataframe[("images", "url")].apply(
            self.get_mime_type
        )

        return dataframe[dataframe[("images", "mime_type")] == self.mime_type]
```



By doing this, we create a custom component that inherits from a `PandasTransformComponent`. This specialised component works with pandas dataframes, allowing us the freedom to modify them as needed before returning the resulting dataframe.
In this particular example, our component guesses the MIME type of each image based on its URL. Subsequently, it adds this information to the `images` subset of the dataframe and returns the filtered dataset based on the desired MIME type.


### Build the component


To use the component, Fondant must package it into an executable Docker image. To achieve this, we need to define a Dockerfile. You can create this file within the `components/filter_images` folder using the following content:

```
FROM --platform=linux/amd64 python:3.8-slim as base

# System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install git -y

# Install requirements
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Fondant
# This is split from other requirements to leverage caching
ARG FONDANT_VERSION=main
RUN pip3 install fondant[aws,azure,gcp]@git+https://github.com/ml6team/fondant@main

# Set the working directory to the component folder
WORKDIR /component/src

# Copy over src-files
COPY src/ .

ENTRYPOINT ["fondant", "execute", "main"]
```

As part of the Dockerfile build process, we install necessary dependencies. Consequently, we must create a `requirements.txt` file in the `components/filter_images` folder. If your components logic demands custom libraries, you can include them in the requirements.txt file but for this example it can be empty since we don't need any extra libraries.



## Make use of your component

To utilise your component, you can incorporate it into the pipeline created in [this guide](/docs/guides/build_a_simple_pipeline.md). To do this, you'll need to add the following code to the `pipeline.py` file:

```python
# Filter mime type component
filter_mime_type = ComponentOp(
    component_dir="components/filter_file_type",
    arguments={"mime_type": "image/png"}
)
```

We initialise the component from a local path, similar to the generic component. However, in this case, the component will be built entirely based on your local files, as the folders contain additional information beyond the ComponentSpec.

Lastly, we need to make adjustments to the pipeline. The step for downloading images can be network-intensive since it involves actual downloads. As a result, we want to pre-filter the files before proceeding with the downloads. To achieve this, we'll modify the pipeline as follows:

```python
pipeline.add_op(load_from_hf_hub)
pipeline.add_op(filter_mime_type, dependencies=[load_from_hf_hub])
pipeline.add_op(download_images, dependencies=[filter_mime_type])
```

We are inserting our custom component as an intermediary step within our pipeline.

Now, you can execute the pipeline once more and examine the results. The final output should exclusively consist of PNG images.

We have designed the custom component to be easily adaptable. For example, if you wish to filter out JPEG files, you can simply change the argument to `image/jpeg`, and your dataset will be populated with JPEGs instead of PNGs

## You Are Done! 🎉 
We now have a simple pipeline that downloads a dataset from HuggingFace hub and outputs the dataset filtered by PNG images. A possible next step is to create a component that **filters the data based on the aspect ratio**? Or **run a CLIP model on the images to get embeddings**?

Expanding upon the concept of custom component implementation, you have the flexibility to create additional custom components. For example, you can design custom components to filter out NSFW (Not Safe For Work) content or to identify and exclude images containing watermarks.
