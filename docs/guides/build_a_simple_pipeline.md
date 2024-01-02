[//]: # (TODO: Add named reference links to the hub for stable links on this page)

# Building your own pipeline

This guide will teach you how to build and run your own pipeline using components available on 
the Fondant hub.

## Overview

In this guide, we will build a pipeline that downloads images from the 
[fondant-cc-25m](https://huggingface.co/datasets/fondant-ai/fondant-cc-25m) dataset and filters 
them.

It consists of three steps:

* **[load_from_hf_hub](../components/hub.md#load_from_hugging_face_hub#description)**: 
  Loads the dataset containing image urls from the Huggingface hub.
* **[download_images](../components/hub.md#download_images#description)**:
  Downloads images from the image urls. 
* **[filter_language](../components/hub.md#filter_language#description)**:
  Filters the images based on the alt text language

## Setting up the environment

We will be using the [local runner](../runners/local.md) to run this pipelines. To set up your local environment, 
please refer to our [installation](installation.md) documentation.

## Building the pipeline

Start by creating a `pipeline.py` file and adding the following code.

```python
from fondant.pipeline import Pipeline

pipeline = Pipeline(
    name="creative_commons_pipline",
    base_path="./data"
)
```

!!! note "IMPORTANT"

    Make sure the provided base_path already exists.

??? "View a detailed reference of the options accepted by the `Pipeline` class"

    ::: fondant.pipeline.Pipeline.__init__
        handler: python
        options:
            show_source: false

## Adding components

Now it's time to incrementally build our pipeline by adding different execution steps or 
`components`. Components are executable elements of a pipeline that consume and produce data.

You can use two types of components with Fondant:

- **Reusable components**: A bunch of reusable components are available on our 
  [hub](https://fondant.ai/en/latest/components/hub/), which you can easily add to your pipeline.
- **Custom components**: You can also implement your own custom component.

If you want to learn more about components, you can check out the 
[components](../components/components.md) documentation.

### 1. A reusable load component

As a first step, we want to read data into our pipeline. In this case, we will load a dataset 
from the HuggingFace Hub. For this, we can use the reusable 
[load_from_hf_hub](../components/hub.md#load_from_hugging_face_hub#description) component.

We can read data into our pipeline using the `Pipeline.read()` method, which returns a (lazy) 
`Dataset`.

```python
import pyarrow as pa

dataset = pipeline.read(
    "load_from_hf_hub",
    arguments={
        "dataset_name": "fondant-ai/fondant-cc-25m",
        "n_rows_to_load": 100,
    },
    produces={
      "alt_text": pa.string(),
      "url": pa.string(),
      "license_location": pa.string(),
      "license_type": pa.string(),
      "webpage_url": pa.string(),
    }
)
```

We provide three arguments to the `.read()` method:

- The name of the reusable component
- Some arguments to configure the component. Check the component's 
  [documentation](../components/hub.md#load_from_hugging_face_hub#arguments) for the supported arguments
- The schema of the data the component will produce. This is necessary for this specific 
  component since the output is dynamic based on the dataset being loaded. You can see this 
  defined in the component [documentation](../components/hub.md#load_from_hugging_face_hub#inputs_outputs) with 
  `additionalProperties: true` under the produces section.

??? "View a detailed reference of the `Pipeline.read()` method"

    ::: fondant.pipeline.Pipeline.read
        handler: python
        options:
            show_source: false

To test the pipeline, you can execute the following command within the pipeline directory:

```bash
fondant run local pipeline.py
```

The pipeline execution will start, initiating the download of the dataset from HuggingFace.
After the pipeline has completed, you can explore the pipeline result using the fondant explorer:

```bash
fondant explore --base_path ./data
```

You can open your browser at `localhost:8501` to explore the loaded data.

### 2. A reusable transform component

Our pipeline has successfully loaded the dataset from HuggingFace. One of these columns, 
`url`, directs us to the original source of the images. To access and utilise these images 
directly, we must download each of them.

Downloading images is a common requirement across various use cases, which is why Fondant provides 
a reusable component specifically for this purpose. This component is appropriately named 
[download_images](../components/hub.md#download_images#description).

We can add this component to our pipeline as follows:

```python
images = dataset.apply(
    "download_images",
)
```

Looking at the component [documentation](../components/hub.md#download_images#consumes), we can see that 
it expects an `"image_url"` field, which was generated by our previous component. This means 
that we can simply chain the components as-is.

### 3. A reusable transform component with non-matching fields

This won't always be the case though. We now want to filter our dataset for images that contain 
English alt text. For this, we leverage the 
[filter_language](../components/hub.md#filter_language#description) component. Looking at the component 
[documentation](../components/hub.md#filter_language#consumes), we can see that it expects an `"text"` 
field, while we would like to apply it to the `"alt_text"` field in our dataset.

We can easily achieve this using the `consumes` argument, which lets us maps the fields that the 
component will consume. Here we indicate that the component should read the `"alt_text"` field 
instead of the `"text"` field.

```python
english_images = images.apply(
  "filter_language",
  arguments={
    "language": "en"
  },
  consumes={
    "text": "alt_text"
  }
)
```

??? "View a detailed reference of the `Dataset.apply()` method"

    ::: fondant.pipeline.pipeline.Dataset.apply
        handler: python
        options:
            show_source: false

## Inspecting your data

Now, you can proceed to execute your pipeline once more and explore the results. In the explorer, 
you will be able to view the images that have been downloaded.

![explorer](../art/guides/explorer.png?raw=true)

Well done! You have now acquired the skills to construct a simple Fondant pipeline by leveraging 
reusable components. In the [next tutorial](implement_custom_components.md), we'll demonstrate how 
you can customise the pipeline by implementing a custom component.
