# Guide - Build a simple pipeline

We present a walkthrough to build by yourself the pipeline presented in the Getting Started section. Have fun!

**Level**: Beginner </br>
**Time**: 20min </br>
**Goal**: After completing this tutorial with Fondant, you will be able to understand the different elements of a pipeline, build, and execute your first pipeline by using existing components. </br>

**Prerequisite**: Make sure docker compose is installed on your local system

## Overview

The sample pipeline that is going to be built in this tutorial demonstrates how to effectively utilise a creative commons image dataset within a fondant pipeline. This dataset comprises images from diverse sources and is available in various data formats.

The pipeline starts with the initialization of the image dataset sourced from HuggingFace and it proceeds with the downloading of these carefully selected images. Accomplishing these tasks necessitates the use of a pre-built generic component (HuggingFace dataset loading) and a reusable component (image downloading).

## Setting up the environment

To set up your local environment, please refer to our getting started documentation. There, you will find the necessary steps to configure your environment.

## Building the pipeline

Everything begins with the pipeline definition. Start by creating a 'pipeline.py' file and adding the following code.

```python
from fondant.pipeline import ComponentOp, Pipeline
pipeline = Pipeline(
pipeline_name="creative_commons_pipline", # This is the name of your pipeline
base_path="./data" # The directory that will be used to store the data
)
```

All you need to initialise a Fondant pipeline are two key parameters:

- **pipeline_name**: This is a name you can use to reference your pipeline. In this example, we've named it after the creative commons-licensed dataset used in the pipeline.
- **base_path**: This is the base path that Fondant should use for storing artifacts and data. In our case, it's a local directory path. However, it can also be a path to a remote storage bucket provided by a cloud service. Please note that the directory you reference must exist; if it doesn't, make sure to create it.

## Adding components

Now it's time to incrementally build our pipeline by adding different execution steps. We refer to these steps as `Components`. Components are executable elements of a pipeline that consume and produce dataframes. The components are defined by a component specification. The component specification is a YAML file that outlines the input and output data structures, along with the arguments utilised by the component and a reference the the docker image used to run the component.

Fondant offers three distinct component types:

- **Reusable components**: These can be readily used without modification.
- **Generic components**: They provide the business logic but may require adjustments to the component spec.
- **Custom components**: The component implementation is user-dependent.

If you want to learn more about components, you can check out the [components documentation](../components/components.md).

### First component to load the dataset

For every pipeline, the initial step is data initialization. In our case, we aim to load the dataset into our pipeline base from HuggingFace. Fortunately, there is already a generic component available called `load_from_hub`.
This component is categorised as a generic component because the structure of the datasets we load from HuggingFace can vary from one dataset to another. While we can leverage the implemented business logic of the component, we must customise the component spec. This customization is necessary to inform the component about the specific columns it will produce.
To utilise this component, it's time to create your first component spec.
Create a folder `component/load_from_hub` and create a `fondant_component.yaml` with the following content:

```yaml
name: Load from hub
description: Component that loads a dataset from the hub
image: ghcr.io/ml6team/load_from_hf_hub:dev

produces:
  images:
    fields:
      alt+text:
        type: string
      url:
        type: string
      license+location:
        type: string
      license+type:
        type: string
      webpage+url:
        type: string

args:
  dataset_name:
    description: Name of dataset on the hub
    type: str
  column_name_mapping:
    description: Mapping of the consumed hub dataset to fondant column names
    type: dict
  image_column_names:
    description:
      Optional argument, a list containing the original image column names in case the
      dataset on the hub contains them. Used to format the image from HF hub format to a byte string.
    type: list
    default: None
  n_rows_to_load:
    description: Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale
    type: int
    default: None
  index_column:
    description: Column to set index to in the load component, if not specified a default globally unique index will be set
    type: str
    default: None
```

As mentioned earlier, the component spec specifies the data structure consumed and/or produced by the component. In this case, the component solely produces data, and this structure is defined within the `produces` section. Fondant operates with hierarchical column structures. In our example, we are defining a column called images with several subset fields.
Now that we have created the component spec, we can incorporate the component into our python code. The next steps involve initialising the component from the component spec and adding it to our pipeline using the following code:

```python
from fondant.pipeline import ComponentOp

load_component_column_mapping = {
    "alt_text": "images_alt+text",
    "image_url": "images_url",
    "license_location": "images_license+location",
    "license_type": "images_license+type",
    "webpage_url": "images_webpage+url",
}

load_from_hf_hub = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "mrchtr/cc-test",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 100,
    }
)

pipeline.add_op(load_from_hf_hub)
```

Two key actions are taking place here:

1. We create a ComponentOp from the registry, configuring the component with specific arguments. In this process, we override default arguments as needed. If we don't provide an argument override, the default values are used. Notably, we are modifying the dataset to be loaded, specifying the number of rows to load (which can be a small number for testing purposes), and mapping columns from the HuggingFace dataset to columns in our dataframe.

2. The add_op method registers the configured component into the pipeline.

To test the pipeline, you can execute the following command within the pipeline directory:

```
fondant run local pipeline.py
```

The pipeline execution will start, initiating the download of the dataset from HuggingFace.
After the pipeline has completed, you can explore the pipeline result using the fondant explorer:

```
fondant explore --base_path ./data
```

You can open your browser at `localhost:8501` to explore the data and columns of the loaded dataset.

### Add a reusable component

Our pipeline has successfully loaded the dataset from HuggingFace. One of these columns, `image_url`, directs us to the original source of the images. To access and utilise these images directly, we must download each of them.

Downloading images is a common requirement across various use cases, which is why Fondant provides a reusable component specifically for this purpose. This component is appropriately named [`download_images`](https://github.com/ml6team/fondant/tree/main/components/download_images).

We can extend our code to incorporate this component into our pipeline with the following code snippet:

```python
# Download images component
download_images = ComponentOp.from_registry(
    name="download_images",
    arguments={}
)

pipeline.add_op(download_images, dependencies=[load_from_hf_hub])
```

The reusable component requires a specific dataset input format to function effectively. Referring to the ComponentHub documentation, this component downloads images based on the URLs provided in the `image_url` column. Fortunately, the column generated by the first component is already named correctly for this purpose.

Instead of initialising the component from a YAML file, we'll use the method `ComponentOp.from_registry(...)` where we can easily specify the name of the reusable component. This is arguably the simplest way to start using a Fondant component.

Finally, we add the component to the pipeline using the `add_op` method. Notably, we define `dependencies=[load_from_hf_hub]` in this step. This command ensures that we chain both components together. Specifically, the `download_images` component awaits the execution input from the `load_from_hf_hub` component.

Now, you can proceed to execute your pipeline once more and explore the results. In the explorer, you will be able to view the images that have been downloaded.

![explorer](https://github.com/ml6team/fondant/blob/main/docs/art/guides/explorer.png?raw=true)

Well done! You have now acquired the skills to construct a simple Fondant pipeline by leveraging generic and reusable components. In our [upcoming tutorial](../guides//implement_custom_components.md), we'll demonstrate how you can customise the pipeline by implementing a custom component.
