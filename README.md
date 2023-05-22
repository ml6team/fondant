# Fondant

Fondant is a data-centric framework to fine-tune [Foundation Models](https://fsi.stanford.edu/publication/opportunities-and-risks-foundation-models) such as:

- Stable Diffusion
- CLIP
- Large Language Models (LLMs like GPT-3)
- Segment Anything (SAM)
- etc.

Fondant focuses on data preparation to fine-tune these models.

## Installation

Fondant can be installed from source using pip:

```
pip install git+https://github.com/ml6team/fondant.git
```

## Introduction

Fondant is built upon [KubeFlow pipelines](https://www.kubeflow.org/docs/components/pipelines/), a cloud-agnostic and open-source framework built by Google to run machine learning workflows on Kubernetes.

Fondant consists of a pipeline that runs a sequence of components (steps) one after the other. Each step implements one logical piece of a data processing pipeline, like image filtering, image embedding, text deduplication, etc.

A Fondant pipeline always includes a loading component as first component, followed by one or more transform components. The loading component loads some initial seed data, with the transform components making transformations on them (like enrichment, filtering, transformation).

## Fondant Components

To implement a new component, one needs to implement 4 things:

- a component specification (YAML file), which lists the name and description of the component, the input and output subsets of the component, as well as custom arguments (like the batch size to use in case the component embeds images)
- a component implementation (Python script), which implements the core logic of the component in plain Python
- a Docker container image, which packages the component's code
- a requirements.txt file which lists the Python dependencies of the component.

The structure of each component should look like this:

```
src
__init__.py
Dockerfile
requirements.txt
```

The `src` folder should contain the component specification and implementation, both are explained below.

### Component specification

A component specification is a YAML file named `fondant_specification.yaml` that could look like this:

```
name: Image filtering
description: Component that filters images based on desired minimum width and height
image: image_filtering:latest

consumes:
  images:
    fields:
      width:
        type: int16
      height:
        type: int16

args:
  min_width:
    description: Desired minimum width
    type: int
  min_height:
    description: Desired minimum height
    type: int
```

It lists the name, description of the component, the input subsets and output subsets that it expects, as well as custom arguments which are relevant for the core logic of the component.

In the example above, the component expects the `images` subset of the dataset as input with the fields `width` and `height`. It doesn't specify any output subsets, which means that this component won't be adding any new subsets to the dataset. It lists 2 custom arguments (`args`), namely a minimum width and height to filter images.

### Component implementation

A component implementation is a Python script (`main.py`) that implements the core logic of the component. This script should always return a single [Dask dataframe](https://docs.dask.org/en/stable/dataframe.html). 

A distinction is made between 2 components: a loading component, which is always the first one in a Fondant pipeline, and a transform component. A loading component loads some initial data and returns a single Dask dataframe, whereas a transform component takes in a single Dask dataframe as input, does some operations on it and returns another single Dask dataframe as output.

Fondant offers the `FondantLoadComponent` and `FondantTransformComponent` classes that serve as boilerplate. To implement your own component, simply overwrite one of these 2 classes. In the example below, we leverage the `FondantTransformComponent` and overwrite its `transform` method.

```
from typing import Dict

import dask.dataframe as dd

from fondant.component import FondantTransformComponent

class ImageFilterComponent(FondantTransformComponent):
    """
    Component that filters images based on height and width.
    """

    def transform(self, df: dd.DataFrame, args: Dict) -> dd.DataFrame:
        """
        Args:
            df: Dask dataframe
            args: args to pass to the function
        
        Returns:
            dataset
        """
        logger.info("Filtering dataset...")
        min_width, min_height = args.min_width, args.min_height
        filtered_df = df[(df["images_width"] > min_width) & (df["images_height"] > min_height)]

        return filtered_df


if __name__ == "__main__":
    component = ImageFilterComponent()
    component.run()
```

## Components zoo

To do: add ready-made components.

## Pipeline zoo

To do: add ready-made pipelines.

## Examples

Example use cases of Fondant include:

- collect additional image-text pairs based on a few seed images and fine-tune Stable Diffusion
- filter an image-text dataset to only include "count" examples and fine-tune CLIP to improve its counting capabilities

Check out the [examples folder](examples) for some illustrations.

## Contributing

We use [poetry](https://python-poetry.org/docs/) and pre-commit to enable a smooth developer flow. Run the following commands to 
set up your development environment:

```commandline
pip install poetry
poetry install
pre-commit install
```